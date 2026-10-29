// Renders content.json into the LPU-format DOCX. TOC / list-of-figures / list-of-tables are inserted as
// real Word fields by finalize_report.py (markers @@TOC@@, @@LOF@@, @@LOT@@).
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType, ShadingType,
  BorderStyle, HeadingLevel, Footer, PageNumber, NumberFormat, LevelFormat, SimpleField, PageBreak, TabStopType,
} = require('docx');

const B = JSON.parse(fs.readFileSync(path.join(__dirname, 'content.json'), 'utf8'));
const FONT = 'Times New Roman';
const CONTENT_W = 9026;                      // A4 (11906) minus 1" margins
const LINE15 = 360;                          // 1.5 line spacing
const HL = 'yellow';

// ---------- inline markup: **bold**, *italic*, [[placeholder]] -------------------------------------------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const tok = m[0];
    if (tok.startsWith('**')) out.push(...runs(tok.slice(2, -2), { ...base, bold: true }));
    else if (tok.startsWith('[[')) out.push(new TextRun({ text: '[' + tok.slice(2, -2) + ']', highlight: HL, font: FONT, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const body = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE15, after: 120 }, ...opts,
});
const center = (text, opts = {}, run = {}) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { line: LINE15, after: 60 }, children: runs(text, { size: 24, ...run }), ...opts,
});

// equations: italic text with _x / _{xyz} rendered as real subscripts
function eqRuns(text) {
  const out = []; const re = /_(\{[^}]+\}|[A-Za-z0-9]+)/g; let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), italics: true, font: FONT, size: 24 }));
    const sub = m[1].startsWith('{') ? m[1].slice(1, -1) : m[1];
    out.push(new TextRun({ text: sub, italics: true, subScript: true, font: FONT, size: 24 }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), italics: true, font: FONT, size: 24 }));
  return out;
}

// ---------- caption counters ---------------------------------------------------------------------------
let figN = 0, tabN = 0;
function caption(label, n, text) {
  return [new TextRun({ text: `${label} `, bold: true, font: FONT, size: 24 }),
          new SimpleField(`SEQ ${label} \\* ARABIC`, String(n)),
          new TextRun({ text: `: ${text}`, bold: true, font: FONT, size: 24 })];
}

// ---------- table ----------------------------------------------------------------------------------------
const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
const borders = { top: border, bottom: border, left: border, right: border };
function mkTable(b) {
  const nC = b.head.length;
  let fr = b.widths || Array(nC).fill(1 / nC);
  const tot = fr.reduce((a, c) => a + c, 0);
  let cw = fr.map(f => Math.round(CONTENT_W * f / tot));
  cw[cw.length - 1] += CONTENT_W - cw.reduce((a, c) => a + c, 0);
  const al = (i) => ({ l: AlignmentType.LEFT, r: AlignmentType.RIGHT, c: AlignmentType.CENTER }[(b.align || [])[i] || 'l']);
  const cell = (txt, i, head) => new TableCell({
    width: { size: cw[i], type: WidthType.DXA }, borders,
    shading: head ? { fill: 'D9E2F3', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ alignment: head ? AlignmentType.CENTER : al(i), spacing: { line: 240, after: 0 },
      children: runs(txt, { size: 20, bold: head }) })],
  });
  const rows = [new TableRow({ tableHeader: true, cantSplit: true, children: b.head.map((h, i) => cell(h, i, true)) })]
    .concat(b.rows.map(r => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) })));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: cw, rows });
}

// ---------- block renderer -------------------------------------------------------------------------------
const bulletRef = 'bullets';
function render(blocks) {
  const out = [];
  for (const b of blocks) {
    switch (b.t) {
      case 'h1': out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun({ text: b.text, font: FONT })] })); break;
      case 'h2': out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: b.text, font: FONT })] })); break;
      case 'h3': out.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: b.text, font: FONT })] })); break;
      case 'p': out.push(body(b.text)); break;
      case 'bullets':
        for (const it of b.items) out.push(new Paragraph({ numbering: { reference: bulletRef, level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE15, after: 60 }, children: runs(it) }));
        break;
      case 'eq':
        out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: 300, before: 80, after: 160 }, children: eqRuns(b.text) }));
        break;
      case 'pb': out.push(new Paragraph({ children: [new PageBreak()] })); break;
      case 'code': for (const ln of b.text.split('\n')) out.push(new Paragraph({ spacing: { line: 240, after: 0 }, children: [new TextRun({ text: ln, font: 'Courier New', size: 18 })] })); break;
      case 'table':
        tabN += 1;
        out.push(new Paragraph({ style: 'Caption', keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80, line: 276 }, children: caption('Table', tabN, b.caption) }));
        out.push(mkTable(b));
        if (b.note) out.push(new Paragraph({ spacing: { line: 240, before: 40, after: 80 }, children: runs(b.note, { size: 20, italics: true }) }));
        else out.push(new Paragraph({ spacing: { line: 240, after: 100 }, children: [] }));
        break;
      case 'fig':
        figN += 1;
        out.push(new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60, line: 240 },
          children: [new ImageRun({ type: 'png', data: fs.readFileSync(b.file), transformation: { width: b.px_w, height: b.px_h },
            altText: { title: b.caption, description: b.caption, name: path.basename(b.file) } })] }));
        out.push(new Paragraph({ style: 'Caption', alignment: AlignmentType.CENTER, spacing: { after: 160, line: 276 }, children: caption('Figure', figN, b.caption) }));
        break;
      case 'refs':
        for (const r of b.items) out.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { line: LINE15, after: 80 }, indent: { left: 567, hanging: 567 }, children: runs(r) }));
        break;
      default: throw new Error('unknown block ' + b.t);
    }
  }
  return out;
}

// ---------- cover ------------------------------------------------------------------------------------------
const TITLE = 'Predicting Telecom Customer Revenue and Churn with Regression and Classification Models';
const cover = [
  center('', {}, {}), center('', {}, {}),
  center('**PREDICTIVE ANALYSIS**', { spacing: { line: LINE15, after: 120 } }),
  center('**PROJECT REPORT**', { spacing: { line: LINE15, after: 120 } }),
  center('(Project Semester [[January-April 2025: edit to your term]])', { spacing: { line: LINE15, after: 360 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: LINE15, after: 480 }, children: [new TextRun({ text: `(${TITLE})`, bold: true, italics: true, size: 32, font: FONT })] }),
  center('Submitted by', { spacing: { line: LINE15, after: 120 } }),
  center('[[Name of student]]'),
  center('Registration No. [[..................]]'),
  center('Programme and Section [[..................]]'),
  center('Course Code [[..................]]', { spacing: { line: LINE15, after: 360 } }),
  center('Under the Guidance of', { spacing: { line: LINE15, after: 120 } }),
  center('**[[Name of faculty coordinator with U.Id and designation]]**', { spacing: { line: LINE15, after: 480 } }),
  center('**Discipline of CSE/IT**'),
  center('**Lovely School of [[______________________]]**'),
  center('**Lovely Professional University, Phagwara**'),
];

const certificate = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240, line: LINE15 }, children: [new TextRun({ text: 'CERTIFICATE', bold: true, underline: {}, font: FONT, size: 28 })] }),
  body('This is to certify that [[student\'s name]] bearing Registration no. [[..........]] has completed [[Course Code]] project titled, **"' + TITLE + '"** under my guidance and supervision. To the best of my knowledge, the present work is the result of his/her original development, effort and study.'),
  center('', {}), center('', {}),
  new Paragraph({ spacing: { line: LINE15 }, children: runs('**[[Signature and Name of the Supervisor]]**') }),
  new Paragraph({ spacing: { line: LINE15 }, children: runs('**[[Designation of the Supervisor]]**') }),
  new Paragraph({ spacing: { line: LINE15 }, children: runs('**School of [[....................................................]]**') }),
  new Paragraph({ spacing: { line: LINE15 }, children: runs('Lovely Professional University') }),
  new Paragraph({ spacing: { line: LINE15 }, children: runs('Phagwara, Punjab.') }),
  new Paragraph({ spacing: { line: LINE15 }, children: runs('Date: [[..........]]') }),
];
const declaration = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240, line: LINE15 }, children: [new TextRun({ text: 'DECLARATION', bold: true, underline: {}, font: FONT, size: 28 })] }),
  body('I, [[student\'s name]], student of [[Program name]] under CSE/IT Discipline at, Lovely Professional University, Punjab, hereby declare that all the information furnished in this project report is based on my own intensive work and is genuine.'),
  center('', {}), center('', {}),
  new Paragraph({ spacing: { line: LINE15 }, tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }], children: runs('Date: [[..........]]\tSignature: [[..........]]') }),
  new Paragraph({ spacing: { line: LINE15 }, tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }], children: runs('Registration No. [[..................]]\tName of the student: [[..................]]') }),
];
const ack = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240, line: LINE15 }, children: [new TextRun({ text: 'ACKNOWLEDGEMENT', bold: true, underline: {}, font: FONT, size: 28 })] }),
  body('I express my sincere gratitude to [[name of faculty coordinator]], [[designation]], Lovely Professional University, for guidance, constructive feedback and encouragement throughout this project. I also thank the faculty of the [[Lovely School of ........]] for the academic foundation in data analysis and machine learning that made this work possible.'),
  body('I acknowledge IBM, for publishing the Telco Customer Churn sample data used here, and the open-source communities behind Python, pandas, NumPy, scikit-learn, Matplotlib and seaborn, whose tools made the analysis reproducible.'),
  body('Finally, I thank my family and friends for their support. [[Edit this paragraph so that it reflects your own thanks.]]'),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { before: 360, line: LINE15 }, children: runs('[[Name of student]]') }),
];

const listHead = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240, line: LINE15 }, pageBreakBefore: true, children: [new TextRun({ text: t, bold: true, font: FONT, size: 28 })] });
const prelims = [
  ...certificate,
  new Paragraph({ pageBreakBefore: true, children: [] }), ...declaration,
  new Paragraph({ pageBreakBefore: true, children: [] }), ...ack,
  listHead('TABLE OF CONTENTS'), new Paragraph({ children: [new TextRun({ text: '@@TOC@@', font: FONT })] }),
  listHead('LIST OF FIGURES'), new Paragraph({ children: [new TextRun({ text: '@@LOF@@', font: FONT })] }),
  listHead('LIST OF TABLES'), new Paragraph({ children: [new TextRun({ text: '@@LOT@@', font: FONT })] }),
];

const footer = (fmt) => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] });
const page = (fmt, start) => ({ size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }, pageNumbers: { start, formatType: fmt } });

const doc = new Document({
  creator: 'Student', title: TITLE, description: 'Predictive Analysis CA2 project report',
  styles: {
    default: { document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { line: LINE15 } } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 28, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 200, line: 276 }, outlineLevel: 0, keepNext: true } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120, line: 276 }, outlineLevel: 1, keepNext: true } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 24, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 200, after: 80, line: 276 }, outlineLevel: 2, keepNext: true } },
      { id: 'Caption', name: 'caption', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 24, bold: true, font: FONT }, paragraph: { spacing: { line: 276 } } },
      { id: 'TOC1', name: 'toc 1', basedOn: 'Normal', next: 'Normal', run: { size: 24, bold: true, font: FONT }, paragraph: { spacing: { line: 300, before: 80, after: 20 } } },
      { id: 'TOC2', name: 'toc 2', basedOn: 'Normal', next: 'Normal', run: { size: 24, font: FONT }, paragraph: { spacing: { line: 300, after: 0 }, indent: { left: 360 } } },
      { id: 'TableofFigures', name: 'table of figures', basedOn: 'Normal', next: 'Normal', run: { size: 24, font: FONT }, paragraph: { spacing: { line: 300, after: 20 } } },
    ],
  },
  numbering: { config: [{ reference: bulletRef, levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [
    { properties: { page: page(NumberFormat.DECIMAL, 1) }, children: cover },
    { properties: { page: page(NumberFormat.LOWER_ROMAN, 1) }, footers: { default: footer() }, children: prelims },
    { properties: { page: page(NumberFormat.DECIMAL, 1) }, footers: { default: footer() }, children: render(B) },
  ],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(path.join(__dirname, 'report_raw.docx'), buf); console.log('raw docx written', buf.length, 'bytes; figures', figN, 'tables', tabN); });
