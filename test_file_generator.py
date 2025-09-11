import random


def generate_city_file(filename: str, num_cities: int,
                       x_range=(-15, 15), y_range=(-15, 20),
                       length_range=(100, 1000), rate_range=(1, 20)):
    with open(filename, "w") as f:
        f.write(f"{num_cities}\n")
        f.write("0 0\n")
        for i in range(num_cities - 1):
            x = random.randint(*x_range)
            y = random.randint(*y_range)
            length = random.randint(*length_range)
            rate = random.randint(*rate_range)
            if i == num_cities - 2:
                f.write(f"{x} {y} {length} {rate}")
            else:
                f.write(f"{x} {y} {length} {rate}\n")

if __name__ == "__main__":
    generate_city_file("cities_100.txt", num_cities=100)
