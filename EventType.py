import argparse
from tqdm import tqdm
import time
import os
import sys

import ROOT as M
import torch
import pca
from PointNetModels.pointnet2C import PointNet
from pathlib import Path
    
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

class EventClassifierPipeline:

    def __init__(self, model_traced_path, onlyACDVeto=True, random_forest_path=None, lookup_path=None):
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
        If not, use PointNet to discriminate between Compton and Pair.
        """
        if not event:
            return "UN", 1.00
        
        nhits = event.GetNHTs()
                
        if nhits == 0:
            return "UN", 1.00

        # 1.Cut for  Photoelectric effect (PHOT)
        if not nhits > 2:
            return "PH", 0.50  # 'PH' for Photoelectric

        # 2. If not Photoelectric, extract hit data and execute PointNet
        data_input = self.extract_hit_data(event)

        if data_input is None or data_input.shape[2] == 0:
            return "UN", 1.00

        with torch.no_grad():
            logits, _ = self.model(data_input)
            prob = torch.sigmoid(logits).item()

        if logits >= 0:
            return "PA", prob  # 'PA' for Pair Production
        elif logits < 0:
            return "CO", 1.0 - prob  # 'CO' for Compton Scattering

        return "UN", 1.00

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

def main(input_path, output_dir, geometry_name, model_traced, onlyACDVeto=True, rf=None, lookup_path=None, debug=False):

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
    pipeline = EventClassifierPipeline(model_traced, onlyACDVeto=onlyACDVeto, random_forest_path=rf, lookup_path=lookup_path)

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
        default="./PointNetModels/test_torch_model_params_final_26-06.pth",
        help="Path to the PointNet model weights file (.pt)."
    )
    
    parser.add_argument(
        "--disable-onlyacd", 
        action="store_false", 
        dest="only_acd_veto",
        help="Disable the strict ACD-only veto and enable the Random Forest/PCA layer."
    )
    parser.add_argument(
        "-rf", "--random-forest", 
        type=str, 
        default="./RandomForest/vega_model.skops",
        help="Path to the Random Forest model file (.skops, used only if ACD-only veto is disabled)."
    )
    parser.add_argument(
        "-pca", "--pca", 
        type=str, 
        default=None, #"./pca_files",
        help="LookupTable pca."
    )

    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode to print MC true processes into the output file."
    )

    # Parse the arguments from command line
    args = parser.parse_args()

    # Pass the parsed arguments directly to the main function
    main(
        input_path=args.input, 
        output_dir=args.output_dir, 
        geometry_name=args.geometry, 
        model_traced=args.model, 
        onlyACDVeto=args.only_acd_veto, 
        rf=None if args.pca is not None else args.random_forest,
        lookup_path=args.pca,
        debug=args.debug
    )
