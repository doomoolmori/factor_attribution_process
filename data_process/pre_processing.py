from data_process import data_read
from data_process import calculation_rank
from data_process import calculation_pct

if __name__ == "__main__":
    data = data_read.DataRead(universe='korea')
    data.dict_of_rank = calculation_rank.rank_dict_add(data)
    data.adj_ri = calculation_pct.make_adj_ri_df(data.dict_of_pandas['RI'])
    data.adj_pct = data.adj_ri.pct_change().shift(-1)

