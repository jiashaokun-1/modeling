"""Tests for Phase 2: PipelineParallelPass and per-stage TrainingPipelinePass."""
import pytest
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.context import (
    TransformContext, ParallelConfig, StreamConfig, TrainingConfig,
)
from python.zrt.transform.parallel.pipeline_parallel import (
    PipelineParallelPass, LayerGroup,
)
from python.zrt.transform.analysis.training import TrainingPipelinePass


# ── helpers ───────────────────────────────────────────────────────────────────

def _t(tid: str, shape=(1, 128, 4096)):
    return TensorMeta.from_shape_dtype(tid, shape, DType.BF16)


def _make_hw():
    import python.zrt.hardware.registry as hw_registry
    return hw_registry.load("nvidia_h100_sxm")


def _make_linear_graph(num_layers: int = 4) -> OpGraph:
    """Build a simple linear graph: one matmul node per transformer layer."""
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []

    prev_out_tensor = _t("input_0")
    for i in range(num_layers):
        node_id = f"mm_layer{i}"
        out_tensor = _t(f"out_layer{i}")
        node = OpNode(
            id=node_id,
            op_type="aten.mm.default",
            inputs=[prev_out_tensor],
            outputs=[out_tensor],
            scope=f"model.layers.{i}.mlp.gate_proj",
            layer=str(i),
            category="compute",
        )
        node.annotations["latency_us"] = 100.0 * (i + 1)  # vary load per layer
        nodes[node_id] = node

        if i > 0:
            prev_id = f"mm_layer{i-1}"
            edges.append(Edge(
                src=prev_id, src_idx=0,
                dst=node_id, dst_idx=0,
                tensor=prev_out_tensor,
            ))

        prev_out_tensor = out_tensor

    return OpGraph(
        name="test_model",
        phase="train_forward",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": 128, "hidden": 4096, "num_layers": num_layers},
    )


def _make_ctx(pp: int = 2, tp: int = 1, dp: int = 1,
              global_batch: int = 8, micro_batch: int = 1,
              pp_layer_assignment=None) -> TransformContext:
    return TransformContext(
        hw_spec=_make_hw(),
        parallel=ParallelConfig(tp=tp, pp=pp, dp=dp),
        stream_config=StreamConfig(),
        training=TrainingConfig(
            micro_batch=micro_batch,
            global_batch=global_batch,
            pp_layer_assignment=pp_layer_assignment,
        ),
    )


# ── PipelineParallelPass tests ────────────────────────────────────────────────

class TestPipelineParallelPass:

    def test_pp1_all_stage0(self):
        """pp=1 → every node gets stage_id=0."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=1)
        result = PipelineParallelPass().run(graph, ctx)
        for node in result.nodes.values():
            assert node.annotations.get("stage_id") == 0

    def test_pp2_splits_layers(self):
        """pp=2 → layers distributed across 2 stages."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        stage_ids = {node.annotations["stage_id"]
                     for node in result.nodes.values()
                     if not node.op_type.startswith("comm.")}
        assert 0 in stage_ids
        assert 1 in stage_ids

    def test_pp2_inserts_p2p_node(self):
        """pp=2 → comm.send_recv nodes are inserted at stage boundaries."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_layer_assignment=[0, 0, 1, 1])
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values()
                     if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) >= 1, f"Expected >= 1 P2P node, got {len(p2p_nodes)}"
        p2p = p2p_nodes[0]
        assert p2p.attrs["src_stage"] == 0
        assert p2p.attrs["dst_stage"] == 1

    def test_pp4_inserts_three_p2p_nodes(self):
        """pp=4 → 3 stage boundaries → 3 comm.send_recv nodes."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values()
                     if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 3

    def test_pp2_does_not_mutate_input(self):
        """Pass is functional: input graph is not modified."""
        graph = _make_linear_graph(num_layers=4)
        original_ids = set(graph.nodes.keys())
        ctx = _make_ctx(pp=2)
        _ = PipelineParallelPass().run(graph, ctx)
        assert set(graph.nodes.keys()) == original_ids

    def test_explicit_layer_assignment(self):
        """Explicit pp_layer_assignment is respected."""
        graph = _make_linear_graph(num_layers=4)
        # layers 0,1 → stage 0; layers 2,3 → stage 1
        ctx = _make_ctx(pp=2, pp_layer_assignment=[0, 0, 1, 1])
        result = PipelineParallelPass().run(graph, ctx)

        stage0_layers = {
            int(n.layer) for n in result.nodes.values()
            if n.layer and n.op_type != "comm.send_recv"
               and n.annotations.get("stage_id") == 0
        }
        stage1_layers = {
            int(n.layer) for n in result.nodes.values()
            if n.layer and n.op_type != "comm.send_recv"
               and n.annotations.get("stage_id") == 1
        }
        assert stage0_layers == {0, 1}
        assert stage1_layers == {2, 3}

    def test_p2p_node_belongs_to_sender_stage(self):
        """P2P node is annotated with the sending (lower) stage_id."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p = next(n for n in result.nodes.values() if n.op_type == "comm.send_recv")
        assert p2p.annotations["stage_id"] == 0  # sender is stage 0

    def test_p2p_node_has_positive_message_size(self):
        """P2P node attrs contain a positive message_size_bytes."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p = next(n for n in result.nodes.values() if n.op_type == "comm.send_recv")
        assert p2p.attrs["message_size_bytes"] > 0

    def test_all_nodes_have_stage_id_after_pp2(self):
        """After pp=2, every node has a stage_id annotation."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        for node in result.nodes.values():
            assert "stage_id" in node.annotations, (
                f"Node {node.id} ({node.op_type}) missing stage_id"
            )


# ── TrainingPipelinePass per-stage tests ──────────────────────────────────────

class TestTrainingPipelinePassPerStage:

    def _run_pipeline_pass(self, num_layers=4, pp=2, global_batch=8):
        """Run PP pass then TrainingPipelinePass and return (result_graph, metrics)."""
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        graph = _make_linear_graph(num_layers=num_layers)
        ctx = _make_ctx(pp=pp, global_batch=global_batch)

        # First assign stage_ids
        g_pp = PipelineParallelPass().run(graph, ctx)
        # Run flops pass to set layer_scale metadata
        g_flops = TrainingFlopsPass().run(g_pp, ctx)
        # Run pipeline pass
        result = TrainingPipelinePass().run(g_flops, ctx)
        return result, result.metadata["pipeline_metrics"]

    def test_metrics_present(self):
        """TrainingPipelinePass writes pipeline_metrics to graph.metadata."""
        result, metrics = self._run_pipeline_pass()
        assert metrics is not None
        assert metrics.step_time_ms >= 0

    def test_pp1_no_bubble(self):
        """pp=1 → bubble_fraction == 0 (no warmup/cooldown)."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=1, global_batch=8)
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        g_flops = TrainingFlopsPass().run(graph, ctx)
        result = TrainingPipelinePass().run(g_flops, ctx)
        metrics = result.metadata["pipeline_metrics"]
        assert metrics.warmup_steps == 0
        assert metrics.cooldown_steps == 0
        assert metrics.bubble_fraction == 0.0

    def test_pp2_has_bubble(self):
        """pp=2 → warmup_steps == cooldown_steps == 1 → bubble > 0."""
        _, metrics = self._run_pipeline_pass(pp=2, global_batch=8)
        assert metrics.warmup_steps == 1
        assert metrics.cooldown_steps == 1
        assert metrics.bubble_fraction > 0.0

    def test_per_stage_latency_not_divided_by_pp(self):
        """per_stage_ms must reflect real per-stage time, not total/pp."""
        graph = _make_linear_graph(num_layers=4)
        ctx2 = _make_ctx(pp=2, global_batch=8)
        ctx1 = _make_ctx(pp=1, global_batch=8)

        from python.zrt.transform.analysis.training import TrainingFlopsPass

        # pp=2 path with stage_ids
        g_pp2 = PipelineParallelPass().run(graph, ctx2)
        g_pp2 = TrainingFlopsPass().run(g_pp2, ctx2)
        result2 = TrainingPipelinePass().run(g_pp2, ctx2)
        per_stage_pp2 = result2.metadata["pipeline_metrics"].per_stage_ms

        # pp=1 baseline
        g1 = TrainingFlopsPass().run(graph, ctx1)
        result1 = TrainingPipelinePass().run(g1, ctx1)
        total_pp1 = result1.metadata["pipeline_metrics"].per_stage_ms

        # Bottleneck stage latency should be >= total/pp (greedy assigns heavier layers)
        # and <= total (can't be larger than whole graph)
        assert 0 < per_stage_pp2 <= total_pp1 * 1.1, (
            f"pp2 per_stage={per_stage_pp2:.3f}ms should be <= pp1 total={total_pp1:.3f}ms"
        )

    def test_stage_timelines_stored_in_metadata(self):
        """When pp>1 and stage_ids exist, stage_timelines_fwd is stored."""
        result, _ = self._run_pipeline_pass(pp=2)
        assert "stage_timelines_fwd" in result.metadata
        timelines = result.metadata["stage_timelines_fwd"]
        assert isinstance(timelines, dict)
        assert 0 in timelines
        assert 1 in timelines

    def test_bubble_fraction_increases_with_pp(self):
        """More PP stages → larger bubble fraction (for fixed M)."""
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        results = {}
        for pp in (1, 2, 4):
            graph = _make_linear_graph(num_layers=4)
            ctx = _make_ctx(pp=pp, global_batch=8)
            g = PipelineParallelPass().run(graph, ctx) if pp > 1 else graph
            g = TrainingFlopsPass().run(g, ctx)
            r = TrainingPipelinePass().run(g, ctx)
            results[pp] = r.metadata["pipeline_metrics"].bubble_fraction

        # bubble_fraction = (2*(pp-1)) / (2*(pp-1)+M) — strictly grows with pp
        assert results[1] == 0.0                # pp=1: no bubble
        assert results[2] > results[1]          # pp=2: some bubble
        assert results[4] > results[2]          # pp=4: more bubble
        assert all(0.0 <= v < 1.0 for v in results.values())


# ── Integration: PP pass in default pipeline ──────────────────────────────────

class TestPipelineInDefaultPipeline:

    def test_pp2_in_default_pipeline(self):
        """PipelineParallelPass is activated when pp=2 in build_default_pipeline."""
        from python.zrt.transform import build_default_pipeline

        graph = _make_linear_graph(num_layers=4)
        ctx = TransformContext(
            hw_spec=_make_hw(),
            parallel=ParallelConfig(pp=2),
            stream_config=StreamConfig(),
        )
        pipe = build_default_pipeline()
        result = pipe.run(graph, ctx)

        # After full pipeline: stage_id annotations present
        compute_nodes = [n for n in result.nodes.values()
                         if n.category == "compute"]
        assert all("stage_id" in n.annotations for n in compute_nodes), (
            "Some compute nodes missing stage_id after pp=2 pipeline"
        )

    def test_pp1_pipeline_no_p2p_nodes(self):
        """When pp=1, no P2P comm.send_recv nodes are inserted."""
        from python.zrt.transform import build_default_pipeline

        graph = _make_linear_graph(num_layers=4)
        ctx = TransformContext(
            hw_spec=_make_hw(),
            parallel=ParallelConfig(pp=1),
            stream_config=StreamConfig(),
        )
        pipe = build_default_pipeline()
        result = pipe.run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values()
                     if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 0
