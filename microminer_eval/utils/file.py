def read_file(filename: str) -> list[str]:
    with open(filename, "r") as file:
        return file.read().splitlines()

