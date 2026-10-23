from abc import ABC, abstractmethod
import time
import pandas as pd

# Excercise 01
class CSVReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def read(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.file_path)
        return self.df

    def preview(self, n: int =5) -> pd.DataFrame:
        if self.df is not None:
            return self.df.head(n)
        print("Dataset no loaded yet.")
        return pd.DataFrame()

# Excercise 02
class MissingValueStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DropMissing(MissingValueStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

class FillMean(MissingValueStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.select_dtypes(include="number").mean())

class FillMode(MissingValueStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.mode().iloc[0])

# Excercise 03
def log_action(func):
    def wrapper(*args, **kwargs):
        print(f"[LOG] Starting: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[LOG] Finished: {func.__name__}")
        return result
    return wrapper

def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"[TIME] {func.__name__} excecuted in {elapsed_time:.4f} seconds")
        return result
    return wrapper

# Exercise 04
class DataTransform(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class NormalizeColumns(DataTransform):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        return df

class RemoveDuplicates(DataTransform):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

class StandardizeText(DataTransform):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str).str.strip().str.title()
        return df

class TransformFactory:
    @staticmethod
    def create_transform(transform_type: str) -> DataTransform:
        transformers = {
            "normalize": NormalizeColumns(),
            "duplicates": RemoveDuplicates(),
            "text": StandardizeText()
        }
        if transform_type not in transformers:
            raise ValueError(f"Unknown transform type: {transform_type}")
        return transformers[transform_type]

# Exercise05
class DataPipeline(ABC):
    """
    The Template Method defines the skeleton of a data processing algorithm.
    """
    def run(self, output_path: str = None) -> pd.DataFrame:
        df = self.load()
        df = self.clean(df)
        df = self.transform(df)
        if output_path:
            self.save(df, output_path)
        return df

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame, output_path: str):
        pass


class CSVDataPipeline(DataPipeline):
    def __init__(self, file_path: str, missing_strategy: MissingValueStrategy, transform_types: list):
        self.reader = CSVReader(file_path)
        self.missing_strategy = missing_strategy
        self.transform_types = transform_types

    @log_action
    @log_time
    def load(self) -> pd.DataFrame:
        return self.reader.read()

    @log_action
    @log_time
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.missing_strategy.handle(df)

    @log_action
    @log_time
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for t_type in self.transform_types:
            transformer = TransformFactory.create_transform(t_type)
            df = transformer.apply(df)
        return df

    @log_action
    @log_time
    def save(self, df: pd.DataFrame, output_path: str):
        df.to_csv(output_path, index=False)
        print(f"Cleaned data successfully saved to {output_path}")


# --- Example Usage ---
if __name__ == "__main__":
    # Instantiate the pipeline with a strategy and list of transformations
    pipeline = CSVDataPipeline(
        file_path="sample_data.csv",
        missing_strategy=FillMean(),
        transform_types=["normalize", "duplicates", "text"]
    )
    
    # Run the full pipeline
    cleaned_df = pipeline.run(output_path="cleaned_data.csv")