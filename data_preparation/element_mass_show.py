import periodictable as pt
import csv


def save_elements_to_csv(filename="elements_mass.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol", "Atomic Mass (g/mol)"])  # name

        for elem in pt.elements:
            if elem is not None and elem.mass is not None:
                writer.writerow([elem.symbol, elem.mass])

    print(f"saved as {filename}")


if __name__ == "__main__":
    save_elements_to_csv()
