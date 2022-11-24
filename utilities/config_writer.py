import yaml


if __name__ == '__main__':
    content = [
        {
            'images': '1_images',
            'targets': '2_targets',
            'predictions': '3_predictions'
        }
    ]

    with open("../config.yaml", 'w') as yamlfile:
        data = yaml.dump(content, yamlfile)
