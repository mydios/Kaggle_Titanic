import pandas as pd

def create_submission(preds, name=''):
    data = {'PassengerId': list(range(892, 892+len(preds))), 'Survived': preds}
    df = pd.DataFrame(data=data)
    n = name + '_submission.csv' if len(name) > 0 else 'submission.csv'
    df.to_csv(n, columns=['PassengerId', 'Survived'], index=False)