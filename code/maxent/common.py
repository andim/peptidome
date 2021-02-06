model_hierarchy = ['model_ncov', 'model', 'model_nskewdiag', 'model_nskew', 'model_nskewfcov']

def set_aminoacidslabel(ax, aminoacidorder):
    naminoacids = len(aminoacidorder)
    ax.set_xticks(range(naminoacids))
    ax.set_yticks(range(naminoacids))
    ax.set_xticklabels(list(aminoacidorder))
    ax.set_yticklabels(list(aminoacidorder))
