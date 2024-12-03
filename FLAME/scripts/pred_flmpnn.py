from FLAME import flsf_predict
import os
tag='_scaffold'
if __name__ == '__main__':
    targets = ['abs', 'emi', 'plqy', 'e']
    for data_base in ['deep4chem', 'FluoDB']:#
        for model in ['deep4chem', 'FluoDB']:
            for target in targets:
                model_path = f'model/flsf{tag}/{model}_{target}'
                input_file = f'data/{data_base}/{target}_test.csv'
                output_file = f'pred/{data_base}/flsf{tag}_{model}_{target}.csv'
                if not os.path.exists(f'pred/{data_base}/'):
                    os.makedirs(f'pred/{data_base}/')
                flsf_predict(model_path, output_file, input_file=input_file)
