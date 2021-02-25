from sklearn.model_selection import train_test_split


def create_label(cats_list:list, dogs_list:list) -> [list,[list]]:
    '''
    This function generates labels for cats(0) and dogs(1)
    :param cats_list: list of cat audio data files
    :param dogs_list: list of dog audio data files
    :return: list of labels for dogs and cats
    '''
    cats_label = 0
    dogs_label = 1
    cat_y = [cats_label]*len(cats_list)
    dog_y = [dogs_label]*len(dogs_list)

    return cat_y, dog_y

def split_data(cats_list: list, dogs_list: list, test_size_ratio: float ):
    cat_y, dog_y = create_label(cats_list, dogs_list)
    X = cats_list + dogs_list
    Y = cat_y + dog_y
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size_ratio, stratify=Y)

    return X_train, X_test, y_train, y_test



