from dragonfly_automation import utils, stage_interpolation


def test_visitation_manager(raw_hcs_position_list, get_mocked_interface):

    # an interface with positions for a full 96-well plate
    # (to match the positions in raw_hcs_position_list)
    micromanager_interface = get_mocked_interface(num_wells=96, num_sites_per_well=36)

    well_ids_to_visit = ['A1', 'A6', 'A12', 'B12', 'B6', 'B1']    
    visitation_manager = stage_interpolation.StageVisitationManager(
        micromanager_interface, well_ids_to_visit, raw_hcs_position_list
    )

    for _ in range(len(well_ids_to_visit)):
        visitation_manager.go_to_next_well()
        visitation_manager.call_afc()

    # there should be a recorded focusdrive position for each well we visited
    assert set(well_ids_to_visit) == set(visitation_manager.measured_focusdrive_positions.keys())
