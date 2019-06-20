import os.path
import glob
import datetime
from shutil import copyfile


header = """---
layout: post
title: {title}
---
"""

code = """```python
{code}
```
"""

jnb_1 = r"""
{::nomarkdown}
{% jupyter_notebook """
jnb_2 = r"""%}
{:/nomarkdown}
"""

for f in glob.iglob('../code/*/*.md'):
    fdate = datetime.datetime.fromtimestamp(os.path.getmtime(f)).date()
    fname = os.path.splitext(os.path.basename(f))[0]
    print(fname)
    fout = '_posts/%s-%s.md'%(fdate, fname)
    copyfile(f, fout)

    fdir = os.path.dirname(f)
    reldir = os.path.basename(fdir)
    with open(fout, 'r') as original:
        title = original.readline()[1:].strip()
        data = original.read()
        data = data.replace('_path_', '/code/{reldir}'.format(reldir=reldir))
    with open(fout, 'w') as modified:
        modified.write(header.format(title=title) + data)

    with open(fout, "a") as foutobj:
        foutobj.write('\n### Code \n')
        for nb in glob.iglob('{}/*.ipynb'.format(fdir)):
            name = os.path.splitext(os.path.basename(nb))[0]
            foutobj.write('#### {name}.ipynb\n\n'.format(name=name))
            #foutobj.write('- [{nbname}.ipynb](/code/{reldir}/{nbname}.ipynb)\n'.format(nbname=nbname,reldir=reldir))
            foutobj.write(jnb_1 + '"/code/{reldir}/'.format(reldir=reldir) + os.path.basename(nb) + '"' + jnb_2)
        for py in glob.iglob('{}/*.py'.format(fdir)):
            pyname = os.path.splitext(os.path.basename(py))[0]
            foutobj.write('#### {pyname}.py\n\n'.format(pyname=pyname))
            with open(py, 'r') as pycodeobj:
                pycode = pycodeobj.read()
                foutobj.write(code.format(code=pycode))
        
    
