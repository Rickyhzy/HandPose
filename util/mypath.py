class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'dynamic_unify':
            # folder that contains class labels
            root_dir = '../data/dynamic_unify'

            # Save preprocess data into output_dir
            output_dir = '../data/output/'

            return root_dir, output_dir

        elif database == 'elas_hand':
            # folder that contains class labels
            root_dir = '../data/dynamic_unify/'

            output_dir = '../data/dynamic_unify/'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/path/to/Models/c3d-pretrained.pth'