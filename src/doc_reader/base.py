from abc import ABC


class BaseDocumentReader(ABC):
    def __init__(self):
        self.file_name = ""
        self.total_pages = 0

    def load_document(self, file_path, category, sub_category):
        pass
