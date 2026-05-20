from pipeline.run_pipeline import run_pipeline

if __name__ == "__main__":
    while True:
        input_query = input("Enter your query: ")
        res = run_pipeline(input_query)
        print(res)
