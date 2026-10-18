"""Read and validate the completed rating worksheet using standard-library ZIP/XML."""
from decimal import Decimal, InvalidOperation
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'

DOCREL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

PKGREL = 'http://schemas.openxmlformats.org/package/2006/relationships'

NS = {'s': MAIN}

HEADERS = ['rating', 'title', 'year', 'title_type', 'sources', 'match_note', 'user_note', 'record_id', 'imdb_url']

def keyed(rows, key, description):
    result = {}
    for row in rows:
        ident = row.get(key, '')
        if not re.fullmatch(r'tt\d+', ident) or ident in result:
            raise ValueError(f'Missing, invalid or duplicate IMDb ID in {description}: {ident!r}')
        result[ident] = row
    return result

def excel_text(text):
    # One pass also preserves an escaped literal such as _x005F_x0041_.
    return re.sub(r'_x([0-9a-fA-F]{4})_', lambda m: chr(int(m[1], 16)), text)

def parse_rating(value, location, allow_blank=False):
    if value == '':
        if allow_blank:
            return None
        raise ValueError(f'Pending blank rating is not permitted in completed data: {location}')
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f'Invalid rating {value!r}: {location}') from error
    if not number.is_finite() or number != number.to_integral_value() or not 0 <= number <= 4:
        raise ValueError(f'Rating must be an integer from 0 through 4: {location}={value!r}')
    return int(number)

def read_rating_xlsx(path, allow_blank):
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheets = [sheet for sheet in workbook.findall('s:sheets/s:sheet', NS) if sheet.attrib.get('name') == 'Ratings']
        if len(sheets) != 1:
            raise ValueError(f'Expected exactly one Ratings worksheet: {path}')
        relationship_id = sheets[0].attrib[f'{{{DOCREL}}}id']
        relations = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = [rel.attrib['Target'] for rel in relations.findall(f'{{{PKGREL}}}Relationship') if rel.attrib['Id'] == relationship_id]
        if len(targets) != 1:
            raise ValueError(f'Missing or ambiguous worksheet relationship: {path}')
        target = targets[0]
        worksheet_path = posixpath.normpath(target.lstrip('/') if target.startswith('/') else posixpath.join('xl', target))
        if worksheet_path.startswith('../'):
            raise ValueError('Invalid workbook worksheet path.')
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            shared = [excel_text(''.join(node.text or '' for node in item.iter(f'{{{MAIN}}}t'))) for item in strings.findall('s:si', NS)]
        sheet = ET.fromstring(archive.read(worksheet_path))

        def cell_value(cell):
            kind = cell.attrib.get('t', 'n')
            raw = cell.findtext('s:v', default='', namespaces=NS)
            if kind == 's':
                return shared[int(raw)]
            if kind == 'inlineStr':
                return excel_text(''.join(node.text or '' for node in cell.iter(f'{{{MAIN}}}t')))
            if kind == 'e':
                raise ValueError(f'Cell error in saved rating worksheet: {path}:{cell.attrib["r"]}={raw}')
            if kind == 'b':
                raise ValueError(f'Boolean is not an accepted rating/identity field: {path}:{cell.attrib["r"]}')
            return excel_text(raw) if kind == 'str' else raw

        rows = {}
        for row in sheet.findall('s:sheetData/s:row', NS):
            index = int(row.attrib['r'])
            cells = {}
            for cell in row.findall('s:c', NS):
                ref = cell.attrib['r']
                column = re.fullmatch(r'([A-Z]+)\d+', ref)[1]
                if column in 'ABCDEFGHI' and len(column) == 1:
                    if index >= 9 and column in {'A', 'G', 'H'} and cell.find('s:f', NS) is not None:
                        raise ValueError(f'Rating, user note and record ID must be saved literal values: {path}:{ref}')
                    cells[column] = cell_value(cell)
            rows[index] = cells
        header = [rows.get(8, {}).get(column, '') for column in 'ABCDEFGHI']
        if header != HEADERS:
            raise ValueError(f'Unexpected rating table schema in {path}: {header}')
        result = []
        for index, cells in sorted(rows.items()):
            if index < 9 or not any(cells.get(column, '') != '' for column in 'ABCDEFGHI'):
                continue
            record = dict(zip(HEADERS, [cells.get(column, '') for column in 'ABCDEFGHI']))
            record['rating'] = parse_rating(record['rating'], f'{path.name}:A{index}', allow_blank)
            record['worksheet_row'] = index
            result.append(record)
        return keyed(result, 'record_id', path.name)
