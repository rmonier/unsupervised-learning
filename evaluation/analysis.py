import subprocess

def main():
    learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
    sigmas = [0.1, 1.0, 2.0]
    epochs = [1, 5, 10]
    verticals = [5, 10, 15, 20]
    horizontals = [5, 10, 15, 20]
    for vertical in verticals:
        for learning_rate in learning_rates:
            for sigma in sigmas:
                for epoch in epochs:
                    for horizontal in horizontals:
                        out_filename = f"som_A3-data_{vertical}_{horizontal}_{learning_rate}_{sigma}_{epoch}.csv"
                        cmd = f"python network/som.py {vertical} {horizontal} \"datasets/preprocessed/A3-data.csv\" {learning_rate} {sigma} \"class\" {epoch} --no-plot -o \"evaluation/results/{out_filename}\""
                        print(f"{cmd}\n")
                        subprocess.run(cmd)
                        print("\n")
                        
                        out_filename = f"som_A3-top10s_{vertical}_{horizontal}_{learning_rate}_{sigma}_{epoch}.csv"
                        cmd = f"python network/som.py {vertical} {horizontal} \"datasets/preprocessed/A3-top10s.csv\" {learning_rate} {sigma} \"top genre\" {epoch} --no-plot -o \"evaluation/results/{out_filename}\""
                        print(f"{cmd}\n")
                        subprocess.run(cmd)
                        print("\n")

if __name__ == '__main__':
    main()