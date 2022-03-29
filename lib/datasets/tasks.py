tasks_voc = {
    "offline":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
    "19-1":
        {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            1: [20],
        },
    "19-1b":
        {
            0: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [5],

        },
    "15-5":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16, 17, 18, 19, 20]

        },

    "15-5s":
            {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                1: [16],
                2: [17],
                3: [18],
                4: [19],
                5: [20]
            },
    "10-2s":
                {
                    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    1: [11, 12],
                    2: [13, 14],
                    3: [15, 16],
                    4: [17, 18],
                    5: [19, 20]
                },
    "10-5-5":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12, 13, 14, 15],
            2: [16, 17, 18, 19, 20]

        },
    "10-10":
            {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            }
}
tasks_coco = {
    "40-40":
        {
            # 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            #     22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
            # 1: [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
            #     67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],

            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            1: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],

        }
}


def get_task_labels(dataset, name, step):
    if dataset == 'pascal_voc':
        task_dict = tasks_voc[name]
    elif dataset == 'coco':
        task_dict = tasks_coco[name]
    else:
        raise NotImplementedError
    # assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"
    labels = list(task_dict[step])
    labels_old = [label for s in range(step) for label in task_dict[s]]
    return labels, labels_old


def get_per_task_classes(dataset, name, step):
    if dataset == 'voc':
        task_dict = tasks_voc[name]
    else:
        raise NotImplementedError
    # assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"
    classes = [len(task_dict[s]) for s in range(step+1)]
    return classes
