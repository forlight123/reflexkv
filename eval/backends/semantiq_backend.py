import os

from eval.backends.vllm_backend import VllmBackend


class SemantiqBackend(VllmBackend):
    name = "semantiq"

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--semantiq-profile", default="default")
        parser.add_argument("--semantiq-config", default=None)
        parser.add_argument("--semantiq-prior-path", default=None)
        parser.add_argument("--semantiq-segment-enable", action="store_true")
        parser.add_argument("--semantiq-segment-page-size", type=int, default=16)
        parser.add_argument(
            "--semantiq-segment-similarity-threshold", type=float, default=0.8
        )
        parser.add_argument("--semantiq-segment-output", default=None)
        parser.add_argument("--semantiq-fake-quant-enable", action="store_true")
        parser.add_argument("--semantiq-fake-quant-seed", type=int, default=0)
        parser.add_argument(
            "--semantiq-quant-method",
            type=int,
            choices=(0, 1),
            default=1,
        )
        return super().add_cli_args(parser)

    def build_engine_args(self, args):
        engine_args = super().build_engine_args(args)
        return self.apply_semantiq_overrides(engine_args, args)

    def _semantiq_cudagraph_none_mode(self, compilation_config):
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
        if cudagraph_mode is not None and not isinstance(cudagraph_mode, str):
            return cudagraph_mode.__class__.NONE
        try:
            from vllm.config import CUDAGraphMode

            return CUDAGraphMode.NONE
        except Exception:
            return "NONE"

    def _disable_cudagraphs_for_semantiq_capture(self, engine_args):
        compilation_config = None
        if isinstance(engine_args, dict):
            compilation_config = engine_args.get("compilation_config")
        else:
            compilation_config = getattr(engine_args, "compilation_config", None)

        if compilation_config is None:
            return

        none_mode = self._semantiq_cudagraph_none_mode(compilation_config)
        if isinstance(compilation_config, dict):
            compilation_config["cudagraph_mode"] = none_mode
            compilation_config["max_cudagraph_capture_size"] = 0
            compilation_config["cudagraph_capture_sizes"] = []
            return

        compilation_config.cudagraph_mode = none_mode
        compilation_config.max_cudagraph_capture_size = 0
        compilation_config.cudagraph_capture_sizes = []

    def apply_semantiq_overrides(self, engine_args, args):
        fake_quant_enabled = getattr(args, "semantiq_fake_quant_enable", False)
        segment_capture_enabled = getattr(args, "semantiq_segment_enable", False)
        env_updates = {
            "SEMANTIQ_QUERY_SEGMENTS_ENABLE": "1"
            if segment_capture_enabled or fake_quant_enabled
            else None,
            "SEMANTIQ_QUERY_SEGMENTS_PAGE_SIZE": str(
                getattr(args, "semantiq_segment_page_size", 16)
            ),
            "SEMANTIQ_QUERY_SEGMENTS_SIMILARITY_THRESHOLD": str(
                getattr(args, "semantiq_segment_similarity_threshold", 0.8)
            ),
            "SEMANTIQ_QUERY_SEGMENTS_OUTPUT_PATH": getattr(
                args, "semantiq_segment_output", None
            ),
            "SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH": getattr(
                args, "semantiq_prior_path", None
            ),
            "SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE": "1"
            if fake_quant_enabled
            else None,
            "SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_SEED": str(
                getattr(args, "semantiq_fake_quant_seed", 0)
            ),
            "SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_METHOD": str(
                getattr(args, "semantiq_quant_method", 1)
            ),
        }

        for key, value in env_updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        if segment_capture_enabled or fake_quant_enabled:
            self._disable_cudagraphs_for_semantiq_capture(engine_args)

        return engine_args
