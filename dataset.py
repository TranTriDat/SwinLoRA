from datasets import DatasetBuilder
import datasets
import yaml
import os


class ISICDataset(DatasetBuilder):
    
    VERSION = datasets.Version("1.0.0")
    
    def _info(self):
        return datasets.DatasetInfo(
            description="ISIC-2018",
            features=datasets.Features({
                'image': datasets.Value('string'),
                'label': datasets.ClassLabel(names=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]),
            }),
        )

    def _split_generators(self, dl_manager):
        # Point to the directory where your data resides
        extracted_path = os.path.join(dl_manager.manual_dir, "Image")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": extracted_path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath,split):
        # You can adjust this method based on the exact structure of your data and how you want to read it
        for label in ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]:
            label_path = os.path.join(filepath, label)
            for img_name in os.listdir(label_path):
                yield f"{label}_{img_name}", {
                    "image": os.path.join(label_path, img_name),
                    "label": label,
}