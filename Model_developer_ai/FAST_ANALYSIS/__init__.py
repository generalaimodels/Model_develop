import sys 
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[1]
sys.path.append(str(package_root))
from FAST_ANALYSIS.Genaerlised_visualization import (
    plot_tensor_grid_sequence,
    plot_tensor_grid_image,
    Plot_tensor_image
)
from FAST_ANALYSIS.generalise_model_anlysis_plots import (
    list_png_files,
    create_video_from_images,
    # summary,
    extract_keys_values,
    load_model_weights,
    save_model_weights,
    create_directory_structure,
    extract_model_details,
    extract_state_dict_details,
    write_to_json_file,
    summarizemodelforhemanth,
    printmodelsummaryforhemanth,
    plot_model_weights,
    sample_train_Listofstr,
    generate_text,
    generate_text_with_strategies
)
from FAST_ANALYSIS.generalised_data_loader import (
    rgb_print,
    load_and_prepare_dataset_for_hemanth,
    prepare_datasetsforhemanth,
    get_dataset_info_for_hemanth
)
from FAST_ANALYSIS.generalised_load_model1 import (
 AdvancedPipelineForhemanth,
 AiModelForHemanth,
 AdvancedPreProcessForHemanth   
)
from FAST_ANALYSIS.Generalised_fileoperation import (
    convert_files_in_directory_text,
    convert_files_in_directory,
    convert_file,
    create_output_dirs,
    clean_text,
    get_files_with_extensions,
    process_files_txtfile,
    read_file_content,
    list_files_with_extensions,
    process_pdfs_from_csv,
    convert_pdf_to_text,
    split_and_save_text,
    organize_files,
    move_files,
    move_file_with_timestamp,
    find_files,
    convert_files_in_directory_video,
    convert_files_in_directory_image,
    convert_files_in_directory_audio,
    convert_to_png,
    convert_to_mp4,
    convert_to_flac,
    load_and_label_files,
    reformat_txt_files
)
from FAST_ANALYSIS.Generalised_trainer import TrainerForHemanth
from FAST_ANALYSIS.data_loader_features import DatasetDictForHemanth
from FAST_ANALYSIS.jsonfile_visyalization import (
    visualize_data,
    load_json
)
from FAST_ANALYSIS.generalised_pipline_any_task import (
    AudioDataPipeline,
    DataProcessor,
    tokenize_function,
    group_texts,
    fim_transform,
    apply_fim,
    load_and_preprocess_dataset,
    ImageProcessingPipeline,
    DataHemanthGPT2,
    fim_spm,
    ConstantLengthDataset,
    preprocess_text,
    permute,
    ConstantLengthDataset_iter,
    combined_text,
    data_tokenization,
    create_data_loaders,
    AdvancedDatasetProcessor,
)

from FAST_ANALYSIS.Generalised_trainer_wrap import create_advanced_model_trainer
from FAST_ANALYSIS.generalised_decodingstragies import generate_text_generalised
from FAST_ANALYSIS.generalised_loadmodel import AdvancedModelLoader
from FAST_ANALYSIS.Generalised_datacollection import convertfolders_to_txtfolder