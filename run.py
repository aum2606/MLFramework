

from src.pipeline.train_pipeline import training_pipleline



def main():
    """ 
        Running the ml pipeline and starting mlflow ui for experimentation tracking
    """
    run = training_pipleline()

    print("running ml pipeline")


if __name__=="__main__":
    main()