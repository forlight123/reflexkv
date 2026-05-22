import reflex_cuda_test_utils


class _FakeCuda:

    def __init__(self):
        self.selected = None

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 3

    @staticmethod
    def mem_get_info(index):
        free_by_index = {
            0: (64, 1024),
            1: (768, 1024),
            2: (512, 1024),
        }
        return free_by_index[index]

    def set_device(self, index):
        self.selected = index


def test_select_reflex_cuda_test_device_picks_freest_visible_gpu(monkeypatch):
    fake_cuda = _FakeCuda()
    monkeypatch.setattr(reflex_cuda_test_utils.torch, "cuda", fake_cuda)

    device = reflex_cuda_test_utils.select_reflex_cuda_test_device(
        min_free_bytes=256,
    )

    assert fake_cuda.selected == 1
    assert device.type == "cuda"
    assert device.index == 1


def test_select_reflex_cuda_test_device_prefers_gpu6_and_gpu7(monkeypatch):

    class FakePreferredCuda:

        def __init__(self):
            self.selected = None

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 8

        @staticmethod
        def mem_get_info(index):
            free_by_index = {
                0: (4096, 8192),
                6: (1024, 8192),
                7: (2048, 8192),
            }
            return free_by_index.get(index, (512, 8192))

        def set_device(self, index):
            self.selected = index

    fake_cuda = FakePreferredCuda()
    monkeypatch.delenv("SEMANTIQ_REFLEX_TEST_GPU_IDS", raising=False)
    monkeypatch.setattr(reflex_cuda_test_utils.torch, "cuda", fake_cuda)

    device = reflex_cuda_test_utils.select_reflex_cuda_test_device(
        min_free_bytes=256,
    )

    assert fake_cuda.selected == 7
    assert device.index == 7
