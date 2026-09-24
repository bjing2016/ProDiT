from collections import defaultdict
import pandas as pd
import csv

def add_loop(cif, loop):
    cols = []
    rows = []
    esc = False
    for i, line in enumerate(loop):
        if len(line) == 1 and line[0][0] == '_': # is key
            cat, attr = line[0].split('.')
            cols.append(attr)
        else: # is value
                
            if line[0] == ';': # end escape block
                esc = False
                
            elif line[0][0] == ';': # begin escape block    
                rows[-1].append(line[0][1:])
                esc = True
            elif esc:
                rows[-1][-1] += line[0]
            else:
                if rows and len(rows[-1]) < len(cols):
                    rows[-1].extend(line)
                else:
                    rows.append(line)
    
    df = pd.DataFrame(data=rows, columns=cols)
    df = df.convert_dtypes()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    cif[cat] = df
    return
            


def read_mmcif(path):
    cif = defaultdict(dict)
    
    with open(path) as f:
        lines = f.read().strip().split('\n')[1:]
    lines = [line.strip() for line in lines if line]

    loop = []
    
    lines = csv.reader(
        lines, delimiter=' ', quotechar="'", skipinitialspace=True
    )
    
    for i, line in enumerate(lines):
        if line[0] == '#':
            if loop: # end of loop
                add_loop(cif, loop)
                loop = []
            continue
            
        if line[0] == 'loop_':
            if loop: # end of loop
                add_loop(cif, loop)
                loop = []
            continue
            
        if line[0][0] == '_' and len(line) == 2: # normal entry
            if loop: # end of loop
                add_loop(cif, loop)
                loop = []
            
            key, val = line
            cat, attr = key.split('.')
            cif[cat][attr] = val
        else: # abnormal entry - not comment or loop start, so must be loop body
            loop.append(line)
    return cif


