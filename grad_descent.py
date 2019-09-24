def fill_params(params):
    k = 1
    l = 5

    for i in range(params.shape[0]):
        if (l != 60):
            params[i][0] = k
            params[i][1] = l
            l += 5
        else:
            l = 5
            k += 1
            params[i][0] = k
            params[i][1] = l
            l += 5


def check_arr(delta_h, max_filter_size, params):

    for i in range(params.shape[0]):
        if (params[i][0] == delta_h and params[i][1] == max_filter_size):
            return params[i][2]
    print("Not found such element")

def put_to_arr(delta_h, max_filter_size, compact, params):

    for i in range(params.shape[0]):
        if (params[i][0] == delta_h and params[i][1] == max_filter_size):
            params[i][2] = compact



def loc_max(initial_image, x, y, thres_volume):

    cr1_min = 1
    cr1_max = 6
    cr2_min = 5
    cr2_max = 55
    r = [-1, 0, 1]
    R = [-5, 0, 5]
    x1 = x
    y1 = y
    for i in r:
        if (x + i < cr1_min or x + i > cr1_max):
            continue
        for j in R:
            if (y + j < cr2_min or y + j > cr2_max):
                continue
            if (segmentation4pool_(initial_image, x+i, y+j, thres_volume) > segmentation4pool_(initial_image, x, y, thres_volume)):
                if (segmentation4pool_(initial_image, x+i, y+j, thres_volume) > segmentation4pool_(initial_image, x1, y1, thres_volume)):
                    x1 = x + i
                    y1 = y + j
    return x1, y1

def grad_descent(initial_image, x, y, thres_volume):

    while (1):
        ax, ay = loc_max(initial_image, x, y, thres_volume)
        if (ax == x and ay == y):
            break
        else:
            x = ax
            y = ay
            print(x, y, check_arr(x, y))
    return x, y

def max_compact(results, params, x, y):

    for i in range(params.shape[0]):
        if (params[i][0] == x and params[i][1] == y):
            prev = params[i][2]
    print(np.max(np.max(results, axis = 0), axis = 0)[2])
    results_ = np.max(results, axis=0)
    for i in range(results_.shape[0]):
        put_to_arr(results_[i][0], results_[i][1], results_[i][2], params)
    if (prev < np.max(results_, axis=0)[2]):
        k = np.argmax(results_, axis=0)[2]
        x = int(results_[k][0])
        y = int(results_[k][1])
    print('max_compact', x, y)
    return x, y

def grad_descent_multithred(initial_image, x, y, thres_volume, n_threads):
    params = np.zeros((66, 3), dtype=float)
    fill_params(params)
    cr1_min = 1
    cr1_max = 6
    cr2_min = 5
    cr2_max = 55
    #pool = mp.Pool(n_threads)
    while (1):
        arglist = []
        results = []
        r = [-1, 0, 1]
        R = [-5, 0, 5]


        for i in r:
            if (x + i >= cr1_min and x + i <= cr1_max):
                for j in R:
                    if (y + j >= cr2_min and y + j <= cr2_max):
                        arglist.append((initial_image, x + i, y + j, thres_volume, params))


        pool = mp.Pool(n_threads)
        results.append(pool.starmap(segmentation4pool_, arglist))
        pool.close()

        x1, y1 = max_compact(results, params, x, y)
        if (x1 == x and y1 == y):
            break
        else:
            x = x1
            y = y1
            print(x, y, check_arr(x, y, params))


def max_params(matrix, comp):
    for i in range(len(matrix)):
        if (matrix[i][2] == comp):
            break
    return matrix[i][0], matrix[i][1]


if __name__ == '__main__':
    #unittest.main()

    path_initial_image = "E:\\Ceramics\\28-29. S2_1@HP1-2\\29. S2_1@HP2\\02. Reconstructed\\x4\\S2-2@HP2_3.49um__rec_x4_0001.png"
    logfoldername = "E:\\Logs"
    thres_volume = 50

    gray_image = read_slices(path_initial_image)

    #gradient descent multi_thread

    gray_image = gray_image[0:gray_image.shape[0] // 2]
    threads = [2]
    for i in threads:
        start_t = time.clock()
        num_threads = i
        print(num_threads)
        grad_descent_multithred(gray_image, 1, 20, thres_volume, num_threads)
        print('time (minutes)', (time.clock() - start_t) / 60)
        print(' ')