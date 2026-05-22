from eval.backends.base import BaseBackend


def _import_vllm():
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    return LLM, SamplingParams, EngineArgs


class VllmBackend(BaseBackend):
    name = "vllm"

    @classmethod
    def add_cli_args(cls, parser):
        _, _, engine_args_cls = _import_vllm()
        return engine_args_cls.add_cli_args(parser)

    def build_engine_args(self, args):
        _, _, engine_args_cls = _import_vllm()
        return engine_args_cls.from_cli_args(args)

    def _create_llm(self, engine_args):
        llm_cls, _, _ = _import_vllm()
        return llm_cls.from_engine_args(engine_args)

    def build_sampling_params(self, gen_config):
        _, sampling_params_cls, _ = _import_vllm()
        return sampling_params_cls(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            max_tokens=gen_config.max_new_tokens,
            stop=gen_config.stop,
        )

    def build(self, args) -> None:
        engine_args = self.build_engine_args(args)
        self._llm = self._create_llm(engine_args)

    def get_prompt_formatter(self):
        tokenizer = self._llm.get_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            return None

        def format_prompt(prompt):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        return format_prompt

    def generate(self, records, gen_config):
        sampling_params = self.build_sampling_params(gen_config)
        outputs = self._llm.generate(
            [record.prompt for record in records],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(records):
            raise ValueError(
                f"vLLM returned {len(outputs)} outputs for {len(records)} prompts"
            )
        predictions = []
        for record, output in zip(records, outputs):
            text = output.outputs[0].text if output.outputs else ""
            predictions.append(
                {
                    "pred": text,
                    "answers": record.answers,
                    "all_classes": record.all_classes,
                }
            )
        return predictions
