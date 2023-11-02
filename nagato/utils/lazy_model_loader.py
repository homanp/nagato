from decouple import config


class LazyModelLoader:
    def __init__(self, model_name: str = None):
        self._model = None
        self._model_name = model_name

    @property
    def model(self):
        if self._model is None and self._model_name is not None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name_or_path=self._model_name, use_auth_token=config("HF_API_KEY")
            )
        return self._model
