import subprocess

def main():
    activations = ["sigmoid"]
    learning_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    momentums = [0, 0.1, 0.5, 0.7, 0.9]
    epochs = [100, 500, 1000]
    topolgy = [[9,5], [4,6], [9], [4,6,8], [8,3]]
    for activation in activations:
        for learning_rate in learning_rates:
            for momentum in momentums:
                for epoch in epochs:
                    for hidden_layer in topolgy:
                        out_filename = f"som_A3-data_{activation}_{epoch}_{learning_rate}_{momentum}_4-"
                        hl = "[4; "
                        for i in hidden_layer:
                            hl += f"{i}; "
                            out_filename += f"{i}-"
                        hl += "1]"
                        out_filename += "1.csv"
                        cmd = f"python network/som.py \"datasets/preprocessed/A3-data\" \"{hl}\" {activation} {learning_rate} {momentum} {epoch} \"{out_filename}\""
                        print(f"{cmd}\n")
                        subprocess.run(cmd)
                        print("\n")
                        
                        out_filename = f"som_A3-top10s_{activation}_{epoch}_{learning_rate}_{momentum}_9-"
                        hl = "[9; "
                        for i in hidden_layer:
                            hl += f"{i}; "
                            out_filename += f"{i}-"
                        hl += "1]"
                        out_filename += "1.csv"
                        cmd = f"python network/som.py \"datasets/split/A3-top10s\" \"{hl}\" {activation} {learning_rate} {momentum} {epoch} \"{out_filename}\""
                        print(f"{cmd}\n")
                        subprocess.run(cmd)
                        print("\n")

if __name__ == '__main__':
    main()