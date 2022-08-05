from data_process import data_read
from data_process import calculation_rank
from data_process import calculation_pct


class PreProcessing:
    def __init__(self, universe):
        """
        space_set = strategy_space(numbers=len(pre_process.dict_of_rank.keys()))
        and data도 여기서 처리하자
        
        :param universe: 
        """


        self.data = data_read.DataRead(universe=universe)
        try:
            path = self.data.pickle_path
            name = self.data.rank_name
            self.dict_of_rank = data_read.read_pickle(path=path, name=name)
            print('already calculation rank')
        except:
            dict_of_rank = calculation_rank.rank_dict_add(data)
            data_read.save_to_pickle(any_=dict_of_rank, path=path, name=name)
            self.dict_of_rank = data_read.read_pickle(path=path, name=name)
            print('calculation rank')

        try:
            name = 'adj_ri.csv'
            self.adj_ri = data_read.read_csv_(path=path, name=name)
            print('already calculation adj ri')
        except:
            adj_ri = calculation_pct.make_adj_ri_df(
                ri_df=self.data.dict_of_pandas['RI'])
            data_read.save_to_csv(df=adj_ri, path=path, name=name)
            self.adj_ri = data_read.read_csv_(path=path, name=name)
            print('calculation adj ri')
        self.adj_ri = self.adj_ri.set_index('date_')

        try:
            name = 'adj_pct.csv'
            self.adj_pct = data_read.read_csv_(path=path, name=name)
            print('already calculation adj pct')
        except:
            adj_pct = self.adj_ri.pct_change().shift(-1)
            data_read.save_to_csv(df=adj_pct, path=path, name=name)
            self.adj_pct = data_read.read_csv_(path=path, name=name)
            print('calculation adj pct')
        self.adj_pct = self.adj_pct.set_index('date_')


if __name__ == "__main__":
    pre_process = PreProcessing(universe='korea')

    data = data_read.DataRead(universe='korea')
    data.dict_of_rank = calculation_rank.rank_dict_add(data)
    data.adj_ri = calculation_pct.make_adj_ri_df(data.dict_of_pandas['RI'])
    data.adj_pct = data.adj_ri.pct_change().shift(-1)

