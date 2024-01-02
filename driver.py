from functions import *

if __name__ == "__main__":
    
    # Parse IBTrACS data for Cat >= 1
    ibtracs_dict, ibtracs_tracks = get_IBTRACS_dict()
    
    # Parse IBTrACS data for all TCs
    all_ibtracs_dict, all_ibtracs_tracks = get_IBTRACS_dict(min_cat=-7)
        
    # Paths to TRACK data
    track_files = ["track_data/2017_track.dat", "track_data/2017_SH_track.dat", "track_data/2018_track.dat", "track_data/2018_SH_track.dat", "track_data/2019_track.dat"]
    
    # Paths to T-TRACK data
    t_track_files = ["t_track_data/2017_t_track.dat", "t_track_data/2017_SH_t_track.dat", "t_track_data/2018_t_track.dat", "t_track_data/2018_SH_t_track.dat", "t_track_data/2019_t_track.dat"]
    
    # Parse TRACK data
    track_dict, track_tracks = get_TRACK_tracks(track_files)
    
    # Parse T-TRACK data
    t_track_dict, t_track_tracks = get_TRACK_tracks(t_track_files)
    
    # Load DL model
    dl_model = tf.keras.models.load_model("final_whole_world") 
    
    # Parse DL data
    dl_dict, dl_tracks = get_DL_dict(dl_model, from_file=True, long_tracks=True)
    
    # Print numbers for Venn diagram in figure 1a
    t_track_venn = generate_venn_numbers(ibtracs_dict, t_track_dict, dl_dict, 0)
    dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods, tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = t_track_venn
    print("Venn Diagram Numbers for T-TRACK")
    print("T-TRACK:", tr)
    print("IBTrACS:", ib)
    print("DL Model:", dl)
    print("T-TRACK and IBTrACS:", ib_tr)
    print("T-TRACK and DL Model:", dl_tr)
    print("IBTrACS and DL Model:", dl_ib)
    print("All Methods:", all_methods)
    
    # Print numbers for Venn diagram in figure 1b
    t_track_venn = generate_venn_numbers(ibtracs_dict, t_track_dict, dl_dict, 1)
    dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods, tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = t_track_venn
    print("\nVenn Diagram Numbers for T-TRACK (NH)")
    print("T-TRACK:", tr)
    print("IBTrACS:", ib)
    print("DL Model:", dl)
    print("T-TRACK and IBTrACS:", ib_tr)
    print("T-TRACK and DL Model:", dl_tr)
    print("IBTrACS and DL Model:", dl_ib)
    print("All Methods:", all_methods)
    
    # Print numbers for Venn diagram in figure 1c
    t_track_venn = generate_venn_numbers(ibtracs_dict, t_track_dict, dl_dict, 2)
    dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods, tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = t_track_venn
    print("\nVenn Diagram Numbers for T-TRACK (SH)")
    print("T-TRACK:", tr)
    print("IBTrACS:", ib)
    print("DL Model:", dl)
    print("T-TRACK and IBTrACS:", ib_tr)
    print("T-TRACK and DL Model:", dl_tr)
    print("IBTrACS and DL Model:", dl_ib)
    print("All Methods:", all_methods)
    print("")
    
    # Overlapping tracks for hurricane strength TCs (figure 2a)
    overlapping_tracks(t_track_tracks, dl_dict, dl_tracks, ibtracs_tracks)
    
    print("")
    
    # Overalapping tracks for all TCs (figure 2b)
    overlapping_tracks(track_tracks, dl_dict, dl_tracks, all_ibtracs_tracks)
    
    # Get correlations
    venn_section_nums, diff_x_ib_track, diff_y_ib_track, diff_x_track_dl, diff_y_track_dl, diff_x_ib_dl, diff_y_ib_dl, diff_x_all_ib_track, diff_y_all_ib_track, diff_x_all_track_dl, diff_y_all_track_dl, diff_x_all_ib_dl, diff_y_all_ib_dl = get_correlations(ibtracs_tracks, t_track_tracks, dl_tracks)
    
    # Plot correlations for figure 6
    plot_correlations(diff_x_ib_track, diff_y_ib_track, diff_x_track_dl, diff_y_track_dl, diff_x_ib_dl, diff_y_ib_dl, diff_x_all_ib_track, diff_y_all_ib_track, diff_x_all_track_dl, diff_y_all_track_dl, diff_x_all_ib_dl, diff_y_all_ib_dl, 10, "T-TRACK", "t_track_corr.pdf")
    
    # Venn diagram numbers for figure 3
    tr_sec, ib_sec, dl_sec, tr_ib_sec, tr_dl_sec, ib_dl_sec, all_sec = venn_section_nums
    print("\nVenn Diagram Numbers for T-TRACK with constraints")
    print("T-TRACK:", tr_sec)
    print("IBTrACS:", ib_sec)
    print("DL Model:", dl_sec)
    print("T-TRACK and IBTrACS:", tr_ib_sec)
    print("T-TRACK and DL Model:", tr_dl_sec)
    print("IBTrACS and DL Model:", ib_dl_sec)
    print("All Methods:", all_sec)
    
    # Get TC frequencies for figure 4
    ib_freq, t_track_freq, track_freq, dl_freq = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 0)
    ib_freq1, t_track_freq1, track_freq1, dl_freq1 = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 1)
    ib_freq2, t_track_freq2, track_freq2, dl_freq2 = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 2)
    ib_freq3, t_track_freq3, track_freq3, dl_freq3 = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 3)
    ib_freq4, t_track_freq4, track_freq4, dl_freq4 = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 4)
    ib_freq5, t_track_freq5, track_freq5, dl_freq5 = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 5)
    ib_freq6, t_track_freq6, track_freq6, dl_freq6 = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 6)
    ib_freq7, t_track_freq7, track_freq7, dl_freq7 = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 7)
    ib_freq8, t_track_freq8, track_freq8, dl_freq8 = get_frequencies(ibtracs_tracks, t_track_tracks, track_tracks, dl_tracks, 8)
    
    # Collect frequencies by source
    ib_freq_all = [ib_freq, ib_freq1, ib_freq2, ib_freq3, ib_freq4, ib_freq5, ib_freq6, ib_freq7, ib_freq8]
    t_track_freq_all = [t_track_freq, t_track_freq1, t_track_freq2, t_track_freq3, t_track_freq4, t_track_freq5, t_track_freq6, t_track_freq7, t_track_freq8]
    dl_freq_all = [dl_freq, dl_freq1, dl_freq2, dl_freq3, dl_freq4, dl_freq5, dl_freq6, dl_freq7, dl_freq8]
    
    # Plot frequencies in figure 4
    plot_frequencies(ib_freq_all, t_track_freq_all, dl_freq_all)
    
    # Plot TC centres from each source as in figure 5
    plot_all_points(ibtracs_dict, t_track_dict, dl_dict, "T-TRACK Points.pdf")
    
    # Plot TC latitude distribution as in figure 7
    plot_distributions(ibtracs_dict, t_track_dict, dl_dict, "T-TRACK", "t_track_hist.pdf")
    
    print("\nPlotting means for T-TRACK (NH)...")
    mslp_min, mslp_max, wind_min, wind_max, vort850_min, vort850_max, vort700_min, vort700_max, vort600_min, vort600_max, nh_means = plot_means(ibtracs_dict, t_track_dict, dl_dict, "T-TRACK", 1, "t_track_means_nh")
    
    print("\nPlotting means for T-TRACK (SH)...")
    mslp_min, mslp_max, wind_min, wind_max, vort850_min, vort850_max, vort700_min, vort700_max, vort600_min, vort600_max, sh_means = plot_means(ibtracs_dict, t_track_dict, dl_dict, "T-TRACK", 2, "t_track_means_sh")
    
    # ibtracs_dict, ibtracs_tracks = get_IBTRACS_dict(cat_opt=1, min_cat=-7)    
    get_venn_diagram_split_by_cats(all_ibtracs_dict, track_dict, dl_dict)
    
    data, labels, files = load_dl_data("/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_test_filtered")
    positive_cases_by_cat(dl_model, data, files)