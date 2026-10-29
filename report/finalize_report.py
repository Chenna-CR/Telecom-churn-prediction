"""Two-pass finaliser: injects real Word TOC / List-of-Figures / List-of-Tables fields whose cached entries carry
correct page numbers (measured from a LibreOffice render), so the lists look right immediately and can still be
refreshed in Word with right-click > Update Field (or Ctrl+A, F9)."""
import json, os, re, shutil, subprocess, sys, zipfile
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
B = json.load(open(f'{HERE}/content.json'))
SOFFICE = '/mnt/skills/public/docx/scripts/office/soffice.py'
RIGHT_POS = 9026

heads = [(b['t'], b['text']) for b in B if b['t'] in ('h1', 'h2')]
figs = [b['caption'] for b in B if b['t'] == 'fig']
tabs = [b['caption'] for b in B if b['t'] == 'table']


def entry_par(style, text, page, first_instr=None, last=False, level_indent=None):
    x = f'<w:p><w:pPr><w:pStyle w:val="{style}"/><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{RIGHT_POS}"/></w:tabs></w:pPr>'
    if first_instr:
        x += ('<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
              f'<w:r><w:instrText xml:space="preserve"> {first_instr} </w:instrText></w:r>'
              '<w:r><w:fldChar w:fldCharType="separate"/></w:r>')
    x += f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>{page}</w:t></w:r>'
    if last:
        x += '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    return x + '</w:p>'


def build_fields(pages):
    toc = []
    for i, (lvl, text) in enumerate(heads):
        toc.append(entry_par('TOC1' if lvl == 'h1' else 'TOC2', text, pages['toc'][i],
                             first_instr='TOC \\o "1-2" \\h \\z \\u' if i == 0 else None, last=(i == len(heads) - 1)))
    lof = [entry_par('TableofFigures', f'Figure {i+1}: {t}', pages['lof'][i], first_instr='TOC \\h \\z \\c "Figure"' if i == 0 else None, last=(i == len(figs) - 1)) for i, t in enumerate(figs)]
    lot = [entry_par('TableofFigures', f'Table {i+1}: {t}', pages['lot'][i], first_instr='TOC \\h \\z \\c "Table"' if i == 0 else None, last=(i == len(tabs) - 1)) for i, t in enumerate(tabs)]
    return {'@@TOC@@': ''.join(toc), '@@LOF@@': ''.join(lof), '@@LOT@@': ''.join(lot)}


def inject(src, dst, pages):
    fields = build_fields(pages)
    with zipfile.ZipFile(src) as zin:
        items = {n: zin.read(n) for n in zin.namelist()}
    xml = items['word/document.xml'].decode('utf8')
    for marker, repl in fields.items():
        pat = re.compile(r'<w:p>(?:(?!</w:p>).)*?' + re.escape(marker) + r'.*?</w:p>', re.S)
        assert pat.search(xml), marker
        xml = pat.sub(lambda m: repl, xml, count=1)
    def _auto(m):                                          # make the implicit 'auto' line rule explicit (LibreOffice treats a missing rule as exact)
        t = m.group(0)
        return t if 'w:lineRule' in t or 'w:line=' not in t else t[:-2] + ' w:lineRule="auto"/>'
    xml = re.sub(r'<w:spacing[^>]*/>', _auto, xml)
    if 'word/styles.xml' in items:
        items['word/styles.xml'] = re.sub(r'<w:spacing[^>]*/>', _auto, items['word/styles.xml'].decode('utf8')).encode('utf8')
    xml = re.sub(r'<w:highlightCs[^>]*/>', '', xml)      # docx-js emits an element the schema does not allow
    items['word/document.xml'] = xml.encode('utf8')
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
        for n, d in items.items():
            zout.writestr(n, d)


def to_pdf(docx):
    subprocess.run(['python', SOFFICE, '--headless', '--convert-to', 'pdf', '--outdir', os.path.dirname(docx), docx], capture_output=True)
    return docx[:-5] + '.pdf'


def page_texts(pdf):
    out = subprocess.run(['pdftotext', '-layout', pdf, '-'], capture_output=True, text=True).stdout
    return out.split('\f')


norm = lambda s: re.sub(r'\s+', ' ', s).strip().lower()


def measure(pdf):
    pages = page_texts(pdf)
    ntext = [norm(p) for p in pages]
    # first main page = first page (after the lists) that begins with the Introduction heading
    start = next(i for i, t in enumerate(ntext) if t.startswith('1. introduction'))
    res = {'toc': [], 'lof': [], 'lot': []}
    cursor = start
    for lvl, text in heads:
        key = norm(text)[:40]
        j = next(i for i in range(cursor, len(ntext)) if key in ntext[i])
        res['toc'].append(j - start + 1); cursor = j
    cursor = start
    for k, lab, items in (('lof', 'figure', figs), ('lot', 'table', tabs)):
        cursor = start
        for n, t in enumerate(items, 1):
            key = norm(f'{lab} {n}: {t}')[:45]
            # captions may wrap in the PDF text; search with normalised whitespace
            j = next((i for i in range(cursor, len(ntext)) if key in ntext[i]), None)
            if j is None:
                key2 = norm(f'{lab} {n}:')
                j = next(i for i in range(cursor, len(ntext)) if key2 in ntext[i])
            res[k].append(j - start + 1); cursor = max(cursor, j - 1)
    return res, len(pages) - (1 if pages and not pages[-1].strip() else 0), start


if __name__ == '__main__':
    raw = f'{HERE}/report_raw.docx'; tmp = f'{HERE}/report_pass1.docx'; final = f'{HERE}/Predictive_Analysis_CA2_Report.docx'
    zero = {'toc': ['00'] * len(heads), 'lof': ['00'] * len(figs), 'lot': ['00'] * len(tabs)}
    inject(raw, tmp, zero)
    pdf1 = to_pdf(tmp)
    pg, n1, start1 = measure(pdf1)
    inject(raw, final, pg)
    pdf2 = to_pdf(final)
    pg2, n2, start2 = measure(pdf2)
    print('pass1 pages', n1, '| pass2 pages', n2, '| main starts at pdf page', start2 + 1, '| stable:', pg == pg2)
    json.dump(pg2, open(f'{HERE}/page_map.json', 'w'))
    if pg != pg2:
        inject(raw, final, pg2); pdf2 = to_pdf(final); print('re-injected with pass-2 numbers')
