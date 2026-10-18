"""Replace two labels in the existing vector framework without redrawing it."""
from pathlib import Path
import re,zlib,argparse,hashlib,json
def update(source,target):
    raw=Path(source).read_bytes();objects={}
    starts=list(re.finditer(rb'(?m)^(\d+) 0 obj\n',raw))
    for m in starts:
        end=raw.index(b'\nendobj',m.end());objects[int(m[1])]=raw[m.end():end]
    assert len(objects)==13 and b'/Contents 13 0 R' in objects[4]
    body=objects[13];i=body.index(b'stream')+len(b'stream');i+=2 if body[i:i+2]==b'\r\n' else 1
    n=int(re.search(rb'/Length (\d+)',body)[1]);stream=zlib.decompress(body[i:i+n]);blocks=list(re.finditer(rb'BT.*?ET',stream,re.S))
    literals=lambda b:b''.join(re.findall(rb'\(([^)]*)\)',b))
    assert literals(blocks[50][0])==b'360 min' and literals(blocks[51][0])==b'screening step'
    widths=list(map(float,re.search(rb'\[(.*?)\]',objects[9],re.S)[1].split()))
    # The existing PDF has a sparse Widths array: the previously unused digit
    # 2 has width zero. Times New Roman uses equal-width decimal digits.
    # Populate that metric so the new 120 label does not overprint its digits.
    assert widths[ord('2')-32]==0
    widths[ord('2')-32]=widths[ord('0')-32]
    old_widths=re.search(rb'\[(.*?)\]',objects[9],re.S)
    replacement=b'['+b' '.join(f'{v:g}'.encode() for v in widths)+b']'
    objects[9]=objects[9][:old_widths.start()]+replacement+objects[9][old_widths.end():]
    def advance(array):
        total=0
        for m in re.finditer(rb'\(([^)]*)\)|([-+]?\d+(?:\.\d+)?)',array):
            total+=sum(widths[c-32] for c in m[1]) if m[1] is not None else -float(m[2])
        return total
    edits=[]
    for k,text in [(50,b'120 min'),(51,b'decision steps')]:
        b=blocks[k][0];a=re.search(rb'\[(.*?)\] TJ',b,re.S);tm=re.search(rb'1 0 0 1 ([\d.]+) ([\d.]+) Tm',b);size=float(re.search(rb'/F1 ([\d.]+) Tf',b)[1])
        x=float(tm[1]);y=tm[2];old=advance(a[1]);new=sum(widths[c-32] for c in text)
        nx=x+(old-new)*size/2000
        v=b[:a.start()]+b'[('+text+b')] TJ'+b[a.end():]
        v=re.sub(rb'1 0 0 1 [\d.]+ ([\d.]+) Tm',lambda m:b'1 0 0 1 '+f'{nx:.6f}'.encode()+b' '+y+b' Tm',v,1)
        edits.append((blocks[k].start(),blocks[k].end(),v))
    changed=stream
    for a,b,v in reversed(edits):changed=changed[:a]+v+changed[b:]
    packed=zlib.compress(changed)
    objects[13]=b'<<\n/Filter /FlateDecode\n/Length '+str(len(packed)).encode()+b'\n>>\nstream\n'+packed+b'\nendstream'
    out=raw[:starts[0].start()];offsets=[0]
    for k in range(1,14):offsets.append(len(out));out+=f'{k} 0 obj\n'.encode()+objects[k]+b'\nendobj\n'
    xref=len(out);out+=b'xref\n0 14\n0000000000 65535 f \n'
    for v in offsets[1:]:out+=f'{v:010d} 00000 n \n'.encode()
    out+=b'trailer\n<<\n/Size 14\n/Root 3 0 R\n/Info 1 0 R\n>>\nstartxref\n'+str(xref).encode()+b'\n%%EOF\n'
    Path(target).write_bytes(out)
    return dict(input_sha256=hashlib.sha256(raw).hexdigest(),output_sha256=hashlib.sha256(out).hexdigest(),changed_text_blocks=[50,51],added_width_for_previously_unused_digit='2',other_content_stream_bytes_unchanged=True,other_11_pdf_objects_unchanged=True,canvas_and_style_unchanged=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target');p.add_argument('--receipt');a=p.parse_args();r=update(a.source,a.target)
    if a.receipt:Path(a.receipt).write_text(json.dumps(r,indent=2),encoding='utf8')
    print(json.dumps(r))
