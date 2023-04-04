
from settings import Settings

def main():
    settings = Settings()
    settings.load_from_file()
    print(settings.run_prefix)



if __name__ == '__main__':
    main()