
with open('submission_copy.csv') as f:
  lines = f.read().splitlines()
  lines = lines[1:]

# lines = lines[:20]
new_lines = []
new_lines.append('id,label')
for line in lines:
  id_, pred = line.split(',')
  new_line = f'{id_},{float(pred):.6f}'
  new_lines.append(new_line)

with open('submission.csv', 'w') as f:
  f.writelines("\n".join(new_lines))