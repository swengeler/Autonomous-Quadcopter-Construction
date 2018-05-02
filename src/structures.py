import numpy as np

a_big_h = np.array([[[2, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1]],
                    [[1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1]],
                    [[1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1]],
                    [[1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 1]]])

another_big_h = np.array([[[1, 1, 0, 0, 1, 1],
                           [1, 1, 0, 0, 1, 1],
                           [1, 1, 0, 0, 1, 1],
                           [1, 1, 2, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 0, 0, 1, 1],
                           [1, 1, 0, 0, 1, 1],
                           [1, 1, 0, 0, 1, 1]],
                          [[0, 1, 0, 0, 1, 0],
                           [0, 1, 0, 0, 1, 0],
                           [0, 1, 0, 0, 1, 0],
                           [0, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 0],
                           [0, 1, 0, 0, 1, 0],
                           [0, 1, 0, 0, 1, 0],
                           [0, 1, 0, 0, 1, 0]]])

a_big_2 = np.array([[[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                     [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])

the_very_beginning = np.array([[[2, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]])

test = np.array([[[1, 2, 1],
                  [1, 0, 1],
                  [1, 1, 1]]])

test_2 = np.array([[[0, 2, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]]])

test_3 = np.array([[[2, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0, 1],
                    [1, 1, 0, 1, 1, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1]]])

test_3d = np.array([[[1, 1, 1],
                     [1, 2, 1],
                     [1, 1, 1]],
                    [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]])

test_4 = np.array([[[0, 0, 0, 1, 2, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0, 1, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 0, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0]]])

test_disjointed = np.array([[[2, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]],
                            [[1, 0, 1],
                             [0, 0, 0],
                             [1, 0, 1]]])

pyramid = np.array([[[2, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1]],
                    [[0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]]])

tower = np.array([[[0, 0, 0, 0],
                   [0, 1, 2, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]],

                  [[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]],

                  [[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]],

                  [[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]],

                  [[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]],

                  [[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]],

                  [[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]],

                  [[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]]])

thing = np.array([[[2, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]],

                  [[1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]],

                  [[1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0]],

                  [[1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0]],

                  [[1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0]],

                  [[1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0]],

                  [[1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]],

                  [[1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]]])

asymmetry_test = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

                           [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

                           [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])

multi_component_test = np.array([[[2, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]],
                                 [[1, 0, 1],
                                  [1, 0, 1],
                                  [1, 0, 1]],
                                 [[1, 0, 1],
                                  [1, 0, 1],
                                  [1, 0, 1]]])

tower_stilts = np.array([[[2, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1]],

                         [[1, 0, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, 0, 1]],

                         [[1, 0, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, 0, 1]],

                         [[1, 0, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, 0, 1]],

                         [[1, 0, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, 0, 1]],

                         [[1, 0, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, 0, 1]],

                         [[1, 0, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, 0, 1]],

                         [[1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1]]])

alternating_tower = np.array([[[1, 1, 1, 1],
                               [1, 2, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1]],

                              [[1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1]],

                              [[1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 1]],

                              [[1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1]],

                              [[1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 1]],

                              [[1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1]],

                              [[1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 1]],

                              [[1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1]],

                              [[1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1]]])

cycle_test = np.array([[[1, 1, 1, 1, 1],
                        [1, 1, 2, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]]])

irregular_cycle_test = np.array([[[0, 0, 1, 2, 1, 1, 0, 0],
                                  [1, 1, 1, 0, 0, 1, 1, 0],
                                  [1, 0, 0, 0, 0, 0, 1, 1],
                                  [1, 1, 0, 0, 0, 0, 0, 1],
                                  [0, 1, 0, 0, 0, 1, 1, 1],
                                  [0, 1, 1, 1, 1, 1, 0, 0]]])

simple_rectangle_10x10 = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])

simple_rectangle_13x13 = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])

simple_rectangle_15x15 = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])

corner_less_test = np.array([[[2, 1, 1, 1, 1, 1],
                              [1, 1, 0, 0, 1, 1],
                              [1, 1, 0, 0, 1, 1],
                              [1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0, 0]]])

double_hole_test = np.array([[[2, 1, 0, 1, 1],
                              [1, 1, 0, 0, 1],
                              [1, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1],
                              [1, 0, 0, 1, 1],
                              [1, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1]]])

quadruple_hole_test = np.array([[[1, 2, 1, 1, 1, 1, 1],
                                 [1, 0, 0, 1, 0, 0, 1],
                                 [1, 0, 0, 1, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 0, 0, 1, 0, 0, 1],
                                 [1, 0, 0, 1, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1]]])

sextuple_hole_test = np.array([[[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                                [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                                [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                                [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
                                [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                                [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]]])

tower_loop = np.array([[[2, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]],

                       [[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]],

                       [[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]],

                       [[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]],

                       [[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]],

                       [[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]],

                       [[1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1]]])

mixed_corner_test = np.array([[[2, 1, 1, 1, 1, 1],
                               [1, 0, 0, 0, 0, 1],
                               [1, 0, 1, 1, 1, 1],
                               [1, 0, 1, 0, 0, 1],
                               [1, 0, 1, 0, 0, 1],
                               [1, 1, 1, 1, 1, 0]]])

mixed_corner_test_2 = np.array([[[0, 1, 1, 2, 1, 1],
                                 [1, 0, 0, 1, 0, 1],
                                 [1, 0, 0, 1, 0, 1],
                                 [1, 1, 1, 1, 0, 1],
                                 [1, 0, 0, 1, 1, 1],
                                 [0, 1, 1, 1, 0, 0]]])

house = np.array([[[2, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]],

                  [[1, 1, 0, 0, 1, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1]],

                  [[1, 1, 0, 0, 1, 1],
                   [1, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 1, 1]],

                  [[1, 1, 0, 0, 1, 1],
                   [1, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 1, 1]],

                  [[1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1]],

                  [[1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]]])

big_loop = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
                      [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

                     [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

                     [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

                     [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])

multi_loop_test = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
                             [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])

smaller_multi_loop_test = np.array([[[2, 1, 1, 1, 1, 1],
                                     [1, 1, 1, 1, 1, 1],
                                     [1, 1, 0, 0, 1, 1],
                                     [1, 1, 0, 0, 1, 1],
                                     [1, 1, 1, 1, 1, 1],
                                     [1, 1, 1, 1, 1, 1]]])

tower_solid_5x5 = np.array([[[2, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],

                            [[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],

                            [[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],

                            [[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],

                            [[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],

                            [[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]]])

tower_stilts_bigger = np.array([[[1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1],
                                 [1, 1, 2, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1]],

                                [[1, 1, 0, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 1]],

                                [[1, 1, 0, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 1]],

                                [[1, 1, 0, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 1]],

                                [[1, 1, 0, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 1]],

                                [[1, 1, 0, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 1]],

                                [[1, 1, 0, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 1]],

                                [[1, 1, 0, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 1]],

                                [[1, 1, 0, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 1]],

                                [[1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1]]])

multi_layered_tower_stilts = np.array([[[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 1, 1, 0, 0],
                                        [0, 0, 1, 2, 1, 1, 0, 0],
                                        [0, 0, 1, 1, 1, 1, 0, 0],
                                        [0, 0, 1, 1, 1, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]],

                                       [[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]],

                                       [[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]],

                                       [[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]],

                                       [[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]],

                                       [[1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1]],

                                       [[1, 0, 0, 0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 0, 1]],

                                       [[1, 0, 0, 0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 0, 1]],

                                       [[1, 0, 0, 0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 0, 1]],

                                       [[1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1]]])
