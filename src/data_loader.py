from datasets import load_dataset, Dataset

def load_tool_calling_dataset(dataset_name="Salesforce/xlam-function-calling-60k", split="train", start=None, end=None) -> Dataset:
    """
    Loads a tool-calling dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        split (str): The split to load (e.g., "train", "test").
        start (int, optional): The starting index of the slice.
        end (int, optional): The ending index of the slice.

    Returns:
        Dataset: The loaded dataset.
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
        if start is not None and end is not None:
            dataset = dataset.select(range(start, end))
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == '__main__':
    tool_calling_dataset = load_tool_calling_dataset()
    if tool_calling_dataset:
        print("Dataset loaded successfully!")
        print(tool_calling_dataset)
        print("Printing the first example of the training set:")
        print(tool_calling_dataset[0])
