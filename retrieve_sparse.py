import hydra
from omegaconf import DictConfig
import pyterrier as pt


@hydra.main(config_path="config", config_name="sparse_retrieval", version_base="1.3")
def main(config: DictConfig) -> None:
    if not pt.started():
        pt.init()

    # Retrieve BM25 model from the batchretrieve index
    bm25 = pt.BatchRetrieve.from_dataset(config.model.index, config.model.variant, wmodel=config.model.wmodel)

    # Get the test topics (dataframe with columns=['qid', 'query'])
    topics = pt.get_dataset(config.dataset).get_topics(config.topics)
    print("topics:")
    print(f"Length: {len(topics)}")
    print(f"Head:\n{topics.head()}")

    topics = pt.get_dataset(config.dataset).get_topics(config.topics)
    top_ranked_docs = (bm25 % config.k)(topics)
    print("top_ranked_docs:")
    print(f"Length: {len(topics)} topics * {config.k} docs = {len(top_ranked_docs)}")
    print(f"Head:\n{top_ranked_docs.head()}")

    # Write to the sparse_runfile.tsv
    output_file = f"{config.dataset}-{config.topics}-{config.model.wmodel}-top{config.k}.tsv"
    with open(output_file, "w") as f:
        # top_ranked_docs.to_csv("sparse_runfile.csv", sep=" ", header=False, index=False)
        for i, row in top_ranked_docs.iterrows():
            f.write(f"{row['qid']}\tQ0\t{row['docno']}\t{i + 1}\t{row['score']}\tsparse\n")


if __name__ == "__main__":
    main()
