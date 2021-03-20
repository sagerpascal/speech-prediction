from main import main, init

if __name__ == '__main__':
    init()
    for i in [1] + list(range(2, 61, 2)):
        main(k_frames=i)
