import glob, os

files = glob.glob('paper/*.tex') + glob.glob('paper/tables/*.tex')
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace [H] with [htbp] to let LaTeX place figures/tables correctly and avoid huge gaps
    new_content = content.replace(r'\begin{table}[H]', r'\begin{table}[htbp]')
    new_content = new_content.replace(r'\begin{figure}[H]', r'\begin{figure}[htbp]')
    
    if content != new_content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f'Updated [H] -> [htbp] in {f}')
