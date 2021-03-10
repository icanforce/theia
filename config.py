from configparser import ConfigParser
import os
config_file = "config.ini"
data_set = ('SYNTHETIC','BRCA', 'LUAD', 'UCEC')
#----------------------------------------------------------------
class Config:
    data_dir = None
    results_dir = None
    
    num_module = None
    thCnt = None
    gene_source = None
    gene_key = None
    gt_source = None
    putative_source = None
    putative_save = None
    ppi_source = None
    ppi_save = None
    mirna_exp_source = None
    mrna_exp_source = None
    mirna_save = None
    mrna_save = None
    selected_exp_save = None
    all_data_save = None
    bayOpt_result = None
    encoder_save = None
    model_save = None
    results = None
    key = None
    save_enable = None

    # Used only for synthetic data case.
    num_sample = None
    num_mirna = None
    num_mrna = None
    num_items_per_module = None
    num_putative_per_mirna = None
    signal_strength = None
    variance_mirna = None
    variance_mrna = None
    

    @staticmethod
    def select_menu(prefix = ''):
       
        default = '0'
        while True:
            print("-------------------------------")
            print(" 0 - Synthetic Data")
            print(" 1 - TCGA-BRCA")
            print(" 2 - TCGA-LUAD")
            print(" 3 - TCGA-UCEC")
            print("-------------------------------")        
            t = input("Enter a number to choose data (default="+default+"): ") or default
        
            try:
                idx = int(t)
                key = data_set[idx]
                print(key + ' selected')
                Config.key = key
                break
            except:
                print("\n\nError : Please enter a valid number!!\n\n") 


        c = ConfigParser()
        c.read(config_file)
        
        Config.data_dir = c.get('DIR', 'data_dir')
        Config.results_dir = prefix + c.get('DIR', 'results_dir')
        if not os.path.exists(Config.results_dir):
            os.mkdir(Config.results_dir)
        
        if key == 'SYNTHETIC':
            Config.num_sample = int( c.get(key, 'num_sample') )
            Config.num_mirna = int( c.get(key, 'num_mirna') )
            Config.num_mrna = int( c.get(key, 'num_mrna') )
            Config.num_module = int( c.get(key, 'num_module') )
            Config.num_items_per_module = int( c.get(key, 'num_items_per_module') )
            Config.num_putative_per_mirna = int( c.get(key, 'num_putative_per_mirna') )
            Config.signal_strength = float( c.get(key, 'signal_strength') )
            Config.variance_mirna = float( c.get(key, 'variance_mirna') )
            Config.variance_mrna = float( c.get(key, 'variance_mrna') )
        else:
            Config.num_module = int( c.get(key, 'num_module') )
            Config.thCnt = int( c.get(key, 'thCnt') )
            Config.gene_source = os.path.join( Config.data_dir, c.get(key, 'gene_source') )            
            Config.gene_key = os.path.join( Config.data_dir, c.get(key, 'gene_key') )            
            Config.gt_source = os.path.join( Config.data_dir, c.get(key, 'gt_source') )            
            
            Config.ppi_source = os.path.join( Config.data_dir, c.get(key, 'ppi_source') )
            Config.ppi_save = os.path.join( Config.results_dir, c.get(key, 'ppi_save') ) 
            
            s = c.get(key, 'putative_source').split(',')
            Config.putative_source = [os.path.join(Config.data_dir, ss) for ss in s]
            Config.putative_save = os.path.join( Config.results_dir, c.get(key, 'putative_save') )
            
            Config.mirna_exp_source = os.path.join( Config.data_dir, c.get(key, 'mirna_exp_source') )
            Config.mrna_exp_source = os.path.join( Config.data_dir, c.get(key, 'mrna_exp_source') )

        Config.mirna_save = os.path.join( Config.results_dir, c.get(key, 'mirna_save') )
        Config.mrna_save = os.path.join( Config.results_dir, c.get(key, 'mrna_save') )
        Config.selected_exp_save = os.path.join( Config.results_dir, c.get(key, 'selected_exp_save') )
        Config.all_data_save = os.path.join( Config.results_dir, c.get(key, 'all_data_save') )
        if not os.path.exists(os.path.join(Config.results_dir, key)):
            os.mkdir(os.path.join(Config.results_dir, key))        
        Config.model_save = os.path.join( Config.results_dir, key, c.get(key, 'model_save') )
        Config.results = os.path.join( Config.results_dir, c.get(key, 'results') )
        Config.save_enable = c.get(key, 'save_enable')

        return
#----------------------------------------------------------------
if __name__ == '__main__':  
    Config.select_menu()
    print(Config.thCnt)
    print(Config.gt_source)
    print(Config.putative_source)
    print(Config.putative_save)
    print(Config.ppi_source)
    print(Config.ppi_save)



