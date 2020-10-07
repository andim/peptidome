model_hierarchy = ['model_ncov', 'model', 'model_nskewdiag', 'model_nskew', 'model_nskewfcov']

labels = {'test': 'test set',
          'train': 'training set',
          'model': '2-point',
          'model_ncov':'2nd moment',
          'independent' : '1-point',
          'model_nskew': '3rd moment',
          'model_nskewdiag': '3rd moment diag',
          'model_nskewfcov' : '3rd moment, 2-point'}

def set_aminoacidslabel(ax, aminoacidorder):
    naminoacids = len(aminoacidorder)
    ax.set_xticks(range(naminoacids))
    ax.set_yticks(range(naminoacids))
    ax.set_xticklabels(list(aminoacidorder))
    ax.set_yticklabels(list(aminoacidorder))
