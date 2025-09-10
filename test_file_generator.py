import random


def generate_city_file(filename: str, num_cities: int,
                       x_range=(0, 30), y_range=(0, 30),
                       length_range=(100, 1000), rate_range=(1, 20)):
    with open(filename, "w") as f:
        # First line: number of cities
        f.write(f"{num_cities}\n")

        # Starting city always at 0 0
        f.write("0 0\n")

        # Generate the rest of the cities
        for i in range(num_cities - 1):
            x = random.randint(*x_range)
            y = random.randint(*y_range)
            length = random.randint(*length_range)
            rate = random.randint(*rate_range)
            if i == num_cities - 2:
                f.write(f"{x} {y} {length} {rate}")  # last line, no newline
            else:
                f.write(f"{x} {y} {length} {rate}\n")

if __name__ == "__main__":
    # Example usage: generate 5 cities
    generate_city_file("cities_10.txt", num_cities=10)
