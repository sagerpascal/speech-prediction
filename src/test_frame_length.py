from main import main, init

if __name__ == '__main__':
    init()
    for i in [1] + list(range(3, 91, 3)):
        main(n_frames=60, k_frames=i, window_shift=150)
