import argparse
from tqdm import tqdm
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ROOT as M
import torch
import pca
from pathlib import Path
    
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

class EventClassifierPipeline:

    def __init__(self, model_traced_path, onlyACDVeto=True, random_forest_path=None, lookup_path=None, three_class=False, min_size=3):
        self.three_class = three_class
        self.min_size = min_size

        if three_class:
            print("Using 3-class PointNet model (PointNetModels/pointnet3C.py)")
            from PointNetModels.pointnet3C import PointNet
        else:
            print("Using 2-class PointNet model (PointNetModels/pointnet2C.py)")
            from PointNetModels.pointnet2C import PointNet

        if not onlyACDVeto:
            if random_forest_path is not None:
                print(f"Loading RF model from {random_forest_path}...")
                self.pca_classifier = pca.VegaClassifier(random_forest_path)
            elif lookup_path is not None:
                print(f"Loading lookup table from {lookup_path}...")
                self.pca_classifier = pca.SimpleClassifier(lookup_path)
            else:
                print("No BKG classifier provided — events will return 0.0")
                self.pca_classifier = None
        else:
            self.pca_classifier = None

        print(f"Loading TorchScript model from {model_traced_path}...")
        self.model = PointNet(add_nhits=False)
        state_dict = torch.load(model_traced_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()  

    def extract_hit_data(self, event, detId=None):      
        nhits = event.GetNHTs()
        if nhits == 0:
            return None

        data = torch.zeros([1, 4, nhits])
        if detId is None:
            for i in range(nhits):
                data[0, 0, i] = event.GetHTAt(i).GetPosition().X()
                data[0, 1, i] = event.GetHTAt(i).GetPosition().Y()
                data[0, 2, i] = event.GetHTAt(i).GetPosition().Z()
                data[0, 3, i] = event.GetHTAt(i).GetEnergy()
            return data 
        else:
            j=0
            for i in range(nhits):
                if event.GetHTAt(i).GetDetectorType() == detId:
                    data[0, 0, j] = event.GetHTAt(i).GetPosition().X()
                    data[0, 1, j] = event.GetHTAt(i).GetPosition().Y()
                    data[0, 2, j] = event.GetHTAt(i).GetPosition().Z()
                    data[0, 3, j] = event.GetHTAt(i).GetEnergy()
                    j += 1
            return data[:, :, :j] if j > 0 else None
    

    def signal_background_classifier(self, event, onlyACDVeto=True, thr=0.99):
        """First layer: Separates signal from background"""
        
        if not event:
            return "UN", 1.00
            
        nhits = event.GetNHTs()
        if nhits == 0:
            return "UN", 1.00

        nhits_ACD  = 0
        for i in range(nhits):
            if event.GetHTAt(i).GetDetectorType() == 4:
                nhits_ACD += 1
        if nhits_ACD > 0:  
            return "MU", 0.99  

        if not onlyACDVeto:
            data = self.extract_hit_data(event, 1)
            prob = pca.analyze(data, event.GetTotalEnergyDeposit(), rf=self.pca_classifier, thr=thr)
            if prob > 0.5:
                return "SIGNAL", prob
            return "MU", 1.-prob
        else:
            return "SIGNAL", 1.00

    def type_of_signal(self, event):
        """Second layer: Checks if the event is a Photoelectric effect.
        If not, use PointNet to discriminate the event topology:
        - binary model  -> Compton vs Pair
        - 3-class model -> Compton vs Pair vs PH
        """
        if not event:
            return "UN", 1.00
        
        nhits = event.GetNHTs()
                
        if nhits == 0:
            return "UN", 1.00

        # 1.Cut for  Photoelectric effect (PHOT)
        if not nhits >= self.min_size:
            return "PH", 0.50  # 'PH' for Photoelectric

        # 2. If not Photoelectric, extract hit data and execute PointNet
        data_input = self.extract_hit_data(event)

        if data_input is None or data_input.shape[2] == 0:
            return "UN", 1.00

        # 3. Dispatch to the head matching the loaded model
        if self.three_class:
            return self._type_of_signal_3class(data_input)
        else:
            return self._type_of_signal_binary(data_input)

    def _type_of_signal_binary(self, data_input):
        """PointNet binary head: single scalar logit -> sigmoid + threshold at 0."""
        with torch.no_grad():
            logits, _ = self.model(data_input)
            prob = torch.sigmoid(logits).item()

        if logits >= 0:
            return "PA", prob  # 'PA' for Pair Production
        elif logits < 0:
            return "CO", 1.0 - prob  # 'CO' for Compton Scattering

        return "UN", 1.00

    def _type_of_signal_3class(self, data_input):
        """PointNet 3-class head: 3 logits (Compton, Pair, Photoelectric) -> softmax + argmax."""
        label_map = {0: "CO", 1: "PA", 2: "PH"}  # same order used in training

        with torch.no_grad():
            logits, _ = self.model(data_input)            # shape [1, 3]
            probs = torch.softmax(logits, dim=1)           # softmax
            pred_idx = torch.argmax(probs, dim=1).item()   # class with highest probability
            prob = probs[0, pred_idx].item()

        return label_map[pred_idx], prob

    def process_event(self, event, onlyACDVeto):
        """Coordinates the sequential execution flow of the cascade pipeline."""
        
        status, prob_bkg = self.signal_background_classifier(event, onlyACDVeto)

        if status == "UN":
            return "UN", 1.00
        if status == "MU":
            return "MU", prob_bkg

        # If it is a good SIGNAL, route it to evaluate the photon type
        final_type, final_prob = self.type_of_signal(event)
        return final_type, final_prob

def main(input_path, output_dir, geometry_name, model_traced, onlyACDVeto=True, rf=None, lookup_path=None, debug=False, three_class=False, min_size=3):

    # Global MEGAlib initialization
    G = M.MGlobal()
    G.Initialize()

    # Load MEGAlib Geometry
    Geometry = M.MDGeometryQuest()
    if Geometry.ScanSetupFile(M.MString(geometry_name)) == True:
        print("Geometry " + geometry_name + " loaded successfully!")
    else:
        print("Unable to load geometry " + geometry_name + " - Aborting!")
        quit()

    # Input file
    path_in = Path(input_path)
    files_to_process = []

    if path_in.is_file():
        files_to_process.append(path_in)
    elif path_in.is_dir():
        files_to_process.extend(path_in.glob("*.sim"))
        files_to_process.extend(path_in.glob("*.sim.gz"))
    else:
        print(f"Error: input path '{input_path}' does not exist or is invalid.. Aborting!")
        quit()

    if not files_to_process:
        print(f"No .sim or .sim.gz files found in '{input_path}'. Exiting.")
        return

    # Initiate the pipeline object
    pipeline = EventClassifierPipeline(model_traced, onlyACDVeto=onlyACDVeto, random_forest_path=rf, lookup_path=lookup_path, three_class=three_class, min_size=min_size)

    path_out_dir = Path(output_dir)
    path_out_dir.mkdir(parents=True, exist_ok=True)

    print("Starting event processing loop...")

    for fn_in in files_to_process:
        base_name = fn_in.name.split('.')[0]
        if path_out_dir.suffix:
            clean_out_dir = path_out_dir.parent
        else:
            clean_out_dir = path_out_dir

        fn_out = clean_out_dir / f"{base_name}.etp"

        print(f"\n[INFO] Processing file: {fn_in.name} -> Target Output: {fn_out}")

        Reader = M.MFileEventsSim(Geometry)
        if Reader.Open(M.MString(str(fn_in))) == False:
            print(f"Unable to open file {fn_in}. Skipping!")
            continue
        
        with open(fn_out, "w") as f_out:

            t_read = t_classify = t_write = 0.0
            i = 0
            with tqdm(desc="Events", unit=" evt") as pbar:
                while True:
                    t0 = time.perf_counter()
                    Event = Reader.GetNextEvent()
                    if not Event:
                        break

                    M.SetOwnership(Event, True)
                    t_read += time.perf_counter() - t0
                    
                    i += 1
                    id_event = Event.GetID()
                
                    t0 = time.perf_counter()
                    event_type, probability = pipeline.process_event(Event, onlyACDVeto)
                    t_classify += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if debug:
                        mc_process = "UNKNOWN"
                        if Event.GetNIAs() > 1:
                            mc_process = str(Event.GetIAAt(1).GetProcess().Data())
                        print(
                            f"SE\nID {id_event}\nMC {mc_process}\nET {event_type}\nTP {probability:.4f}",
                            file=f_out,
                        )
                    else:
                        print(
                            f"SE\nID {id_event}\nET {event_type}\nTP {probability:.4f}",
                            file=f_out,
                        )
                    t_write += time.perf_counter() - t0
                    del Event
                    pbar.update(1)
                        
                    if i % 500 == 0:
                        pbar.set_postfix({
                            "read":     f"{t_read  / i * 1000:.1f}ms",
                            "classify": f"{t_classify / i * 1000:.1f}ms",
                            "write":    f"{t_write / i * 1000:.1f}ms",
                        })
                        f_out.flush()

            print(f"\nDONE. {i} events processed.")
            print(f"  avg read    : {t_read     / i * 1000:.2f} ms/evt")
            print(f"  avg classify: {t_classify / i * 1000:.2f} ms/evt")
            print(f"  avg write   : {t_write    / i * 1000:.2f} ms/evt")
            print(f"[OK] File {fn_in.name} completed successfully. Saved to {fn_out}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Event Classifier Pipeline for MEGAlib simulation files."
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        default="./mini_test.sim.gz",
        help="Path to a single .sim/.sim.gz file OR to a directory containing them."
    )
    parser.add_argument(
        "-o", "--output-dir", 
        type=str, 
        default="./output_etp",
        help="Directory where output .etp files will be saved."
    )
    parser.add_argument(
        "-g", "--geometry", 
        type=str, 
        default="../../simuComPair/Geometry/ComPair_23/ComPair23.geo.setup",
        help="Path to the MEGAlib geometry setup file (.geo.setup)."
    )
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default=None,
        help="Path to the PointNet model weights file (.pt). If omitted, defaults to "
             "the 2-class or 3-class checkpoint depending on whether -3c/--three-class is set."
    )
    parser.add_argument(
        "--disable-onlyacd", 
        action="store_false", 
        dest="only_acd_veto",
        help="Disable the strict ACD-only veto and enable the Random Forest/PCA layer "
             "(implied automatically if -rf or -pca is provided)."
    )
    parser.add_argument(
        "-rf", "--random-forest", 
        type=str, 
        default=None,
        help="Path to the Random Forest model file (.skops). Providing this flag "
             "automatically disables the ACD-only veto."
    )
    parser.add_argument(
        "-pca", "--pca", 
        type=str, 
        default=None, #"./pca_files",
        help="Path to the lookup-table pca file. Providing this flag automatically "
             "disables the ACD-only veto."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode to print MC true processes into the output file."
    )
    parser.add_argument(
        "-3c", "--three-class",
        action="store_true",
        dest="three_class",
        help="Use the 3-class PointNet model (PointNetModels/pointnet3C.py) instead "
             "of the default 2-class model (PointNetModels/pointnet2C.py)."
    )
    parser.add_argument(
        "--msize", "--min-event-size",
        type=int,
        default=3,
        help="Minimum number of hits required for an event to be processed by the PointNet. "
             "Signal Events with fewer hits will be classified as 'PH'."
    )

    # Parse the arguments from command line
    args = parser.parse_args()

    DEFAULT_RF_PATH = "./RandomForest/vega_model.skops"
    DEFAULT_MODEL_PATH_2C = "./PointNetModels/test_torch_model_params_train_2C_nhits2.pth"
    DEFAULT_MODEL_PATH_3C = "./PointNetModels/test_torch_model_params_3C_nhits.pth"

    if args.random_forest is not None and args.pca is not None:
        parser.error(
            "--random-forest/-rf and --pca/-pca are mutually exclusive: "
            "choose only one background classifier."
        )

    # Explicitly asking for -rf or -pca implies you want the veto disabled:
    # no need to also pass --disable-onlyacd.
    if args.random_forest is not None or args.pca is not None:
        args.only_acd_veto = False

    # If the veto is disabled (via --disable-onlyacd alone) but neither -rf nor
    # -pca was given, fall back to the default Random Forest model path.
    rf_path = args.random_forest
    if not args.only_acd_veto and rf_path is None and args.pca is None:
        rf_path = DEFAULT_RF_PATH

    # If -m/--model wasn't given, pick the default checkpoint matching -3c/--three-class.
    model_path = args.model
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH_3C if args.three_class else DEFAULT_MODEL_PATH_2C

    # Pass the parsed arguments directly to the main function
    main(
        input_path=args.input, 
        output_dir=args.output_dir, 
        geometry_name=args.geometry, 
        model_traced=model_path, 
        onlyACDVeto=args.only_acd_veto, 
        rf=None if args.pca is not None else rf_path,
        lookup_path=args.pca,
        debug=args.debug,
        min_size=args.msize,
        three_class=args.three_class
    )
