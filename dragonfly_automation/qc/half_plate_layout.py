'''
Platemaps for canonical 'half-plate' pipeline plate imaging acquisitions

In these acquisitions, half of a pipeline plate is imaged by transposing
either the first or second half of the pipeline plate (columns 1-6 or columns 7-12)
to the region of the imaging plate spanned by B2 to G9. 

The maps hard-coded below encode the way in which this transposition is performed,
and are used in `pipeline_plate_qc.rename_raw_tiffs_from_half_plate`

Keith Cheveralls
November 2019

'''


first_half = [
    {'imaging_well_id': 'B9', 'pipeline_well_id': 'H1'},
    {'imaging_well_id': 'B8', 'pipeline_well_id': 'G1'},
    {'imaging_well_id': 'B7', 'pipeline_well_id': 'F1'},
    {'imaging_well_id': 'B6', 'pipeline_well_id': 'E1'},
    {'imaging_well_id': 'B5', 'pipeline_well_id': 'D1'},
    {'imaging_well_id': 'B4', 'pipeline_well_id': 'C1'},
    {'imaging_well_id': 'B3', 'pipeline_well_id': 'B1'},
    {'imaging_well_id': 'B2', 'pipeline_well_id': 'A1'},

    {'imaging_well_id': 'C2', 'pipeline_well_id': 'A2'},
    {'imaging_well_id': 'C3', 'pipeline_well_id': 'B2'},
    {'imaging_well_id': 'C4', 'pipeline_well_id': 'C2'},
    {'imaging_well_id': 'C5', 'pipeline_well_id': 'D2'},
    {'imaging_well_id': 'C6', 'pipeline_well_id': 'E2'},
    {'imaging_well_id': 'C7', 'pipeline_well_id': 'F2'},
    {'imaging_well_id': 'C8', 'pipeline_well_id': 'G2'},
    {'imaging_well_id': 'C9', 'pipeline_well_id': 'H2'},

    {'imaging_well_id': 'D9', 'pipeline_well_id': 'H3'},
    {'imaging_well_id': 'D8', 'pipeline_well_id': 'G3'},
    {'imaging_well_id': 'D7', 'pipeline_well_id': 'F3'},
    {'imaging_well_id': 'D6', 'pipeline_well_id': 'E3'},
    {'imaging_well_id': 'D5', 'pipeline_well_id': 'D3'},
    {'imaging_well_id': 'D4', 'pipeline_well_id': 'C3'},
    {'imaging_well_id': 'D3', 'pipeline_well_id': 'B3'},
    {'imaging_well_id': 'D2', 'pipeline_well_id': 'A3'},

    {'imaging_well_id': 'E2', 'pipeline_well_id': 'A4'},
    {'imaging_well_id': 'E3', 'pipeline_well_id': 'B4'},
    {'imaging_well_id': 'E4', 'pipeline_well_id': 'C4'},
    {'imaging_well_id': 'E5', 'pipeline_well_id': 'D4'},
    {'imaging_well_id': 'E6', 'pipeline_well_id': 'E4'},
    {'imaging_well_id': 'E7', 'pipeline_well_id': 'F4'},
    {'imaging_well_id': 'E8', 'pipeline_well_id': 'G4'},
    {'imaging_well_id': 'E9', 'pipeline_well_id': 'H4'},

    {'imaging_well_id': 'F9', 'pipeline_well_id': 'H5'},
    {'imaging_well_id': 'F8', 'pipeline_well_id': 'G5'},
    {'imaging_well_id': 'F7', 'pipeline_well_id': 'F5'},
    {'imaging_well_id': 'F6', 'pipeline_well_id': 'E5'},
    {'imaging_well_id': 'F5', 'pipeline_well_id': 'D5'},
    {'imaging_well_id': 'F4', 'pipeline_well_id': 'C5'},
    {'imaging_well_id': 'F3', 'pipeline_well_id': 'B5'},
    {'imaging_well_id': 'F2', 'pipeline_well_id': 'A5'},
    
    {'imaging_well_id': 'G2', 'pipeline_well_id': 'A6'},
    {'imaging_well_id': 'G3', 'pipeline_well_id': 'B6'},
    {'imaging_well_id': 'G4', 'pipeline_well_id': 'C6'},
    {'imaging_well_id': 'G5', 'pipeline_well_id': 'D6'},
    {'imaging_well_id': 'G6', 'pipeline_well_id': 'E6'},
    {'imaging_well_id': 'G7', 'pipeline_well_id': 'F6'},
    {'imaging_well_id': 'G8', 'pipeline_well_id': 'G6'},
    {'imaging_well_id': 'G9', 'pipeline_well_id': 'H6'}
 ]

second_half = [
    {'imaging_well_id': 'B9', 'pipeline_well_id': 'H7'},
    {'imaging_well_id': 'B8', 'pipeline_well_id': 'G7'},
    {'imaging_well_id': 'B7', 'pipeline_well_id': 'F7'},
    {'imaging_well_id': 'B6', 'pipeline_well_id': 'E7'},
    {'imaging_well_id': 'B5', 'pipeline_well_id': 'D7'},
    {'imaging_well_id': 'B4', 'pipeline_well_id': 'C7'},
    {'imaging_well_id': 'B3', 'pipeline_well_id': 'B7'},
    {'imaging_well_id': 'B2', 'pipeline_well_id': 'A7'},

    {'imaging_well_id': 'C2', 'pipeline_well_id': 'A8'},
    {'imaging_well_id': 'C3', 'pipeline_well_id': 'B8'},
    {'imaging_well_id': 'C4', 'pipeline_well_id': 'C8'},
    {'imaging_well_id': 'C5', 'pipeline_well_id': 'D8'},
    {'imaging_well_id': 'C6', 'pipeline_well_id': 'E8'},
    {'imaging_well_id': 'C7', 'pipeline_well_id': 'F8'},
    {'imaging_well_id': 'C8', 'pipeline_well_id': 'G8'},
    {'imaging_well_id': 'C9', 'pipeline_well_id': 'H8'},

    {'imaging_well_id': 'D9', 'pipeline_well_id': 'H9'},
    {'imaging_well_id': 'D8', 'pipeline_well_id': 'G9'},
    {'imaging_well_id': 'D7', 'pipeline_well_id': 'F9'},
    {'imaging_well_id': 'D6', 'pipeline_well_id': 'E9'},
    {'imaging_well_id': 'D5', 'pipeline_well_id': 'D9'},
    {'imaging_well_id': 'D4', 'pipeline_well_id': 'C9'},
    {'imaging_well_id': 'D3', 'pipeline_well_id': 'B9'},
    {'imaging_well_id': 'D2', 'pipeline_well_id': 'A9'},

    {'imaging_well_id': 'E2', 'pipeline_well_id': 'A10'},
    {'imaging_well_id': 'E3', 'pipeline_well_id': 'B10'},
    {'imaging_well_id': 'E4', 'pipeline_well_id': 'C10'},
    {'imaging_well_id': 'E5', 'pipeline_well_id': 'D10'},
    {'imaging_well_id': 'E6', 'pipeline_well_id': 'E10'},
    {'imaging_well_id': 'E7', 'pipeline_well_id': 'F10'},
    {'imaging_well_id': 'E8', 'pipeline_well_id': 'G10'},
    {'imaging_well_id': 'E9', 'pipeline_well_id': 'H10'},

    {'imaging_well_id': 'F9', 'pipeline_well_id': 'H11'},
    {'imaging_well_id': 'F8', 'pipeline_well_id': 'G11'},
    {'imaging_well_id': 'F7', 'pipeline_well_id': 'F11'},
    {'imaging_well_id': 'F6', 'pipeline_well_id': 'E11'},
    {'imaging_well_id': 'F5', 'pipeline_well_id': 'D11'},
    {'imaging_well_id': 'F4', 'pipeline_well_id': 'C11'},
    {'imaging_well_id': 'F3', 'pipeline_well_id': 'B11'},
    {'imaging_well_id': 'F2', 'pipeline_well_id': 'A11'},
    
    {'imaging_well_id': 'G2', 'pipeline_well_id': 'A12'},
    {'imaging_well_id': 'G3', 'pipeline_well_id': 'B12'},
    {'imaging_well_id': 'G4', 'pipeline_well_id': 'C12'},
    {'imaging_well_id': 'G5', 'pipeline_well_id': 'D12'},
    {'imaging_well_id': 'G6', 'pipeline_well_id': 'E12'},
    {'imaging_well_id': 'G7', 'pipeline_well_id': 'F12'},
    {'imaging_well_id': 'G8', 'pipeline_well_id': 'G12'},
    {'imaging_well_id': 'G9', 'pipeline_well_id': 'H12'}
]