from abc import ABC
from src.data_processor._base import BaseFeatureProcessor
import pandas as pd


class Phase2Prob1FeatureProcessor(BaseFeatureProcessor):
    def __init__(self, features=None, categorical_features=None, agg_features=None):
        super().__init__(features=features, 
                         categorical_features=categorical_features)
        if self.FEATURES is None:
            self.FEATURES = ['feature1', 'feature5', 'feature6',
                         'feature7', 'feature8', 'feature9', 'feature10', 'feature11',
                         'feature12', 'feature13', 'feature14', 'feature15', 'feature16',
                         'feature17', 'feature18', 'feature19', 'feature20', 'feature21',
                         'feature22', 'feature23', 'feature24', 'feature25', 'feature26',
                         'feature27', 'feature28', 'feature29', 'feature30', 'feature31',
                         'feature32', 'feature33', 'feature34', 'feature35', 'feature36',
                         'feature37', 'feature38', 'feature39', 'feature40', 'feature41']
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = []

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data:pd.DataFrame)-> pd.DataFrame:
        self.data = data.copy()
        self.one_hot()
        # self.convert_to_categorical()
        out = self.data.copy()
        self.flush()
        return out[self.FEATURES]

    def one_hot(self):
        for col in ['tcp', 'udp', 'gre', 'sccopmce', 'a/n', 'unas', 'emcon', 'crtp',
       'pup', 'stp', 'arp', 'ospf', '3pc', 'mtp', 'i-nlsp', 'iso-ip',
       'any', 'aris', 'snp', 'sctp', 'ptp', 'ipcomp', 'wb-expak', 'dcn',
       'idpr-cmtp', 'tcf', 'netblt', 'trunk-2', 'isis', 'sprite-rpc',
       'dgp', 'chaos', 'nsfnet-igp', 'egp', 'ip', 'iplt', 'aes-sp3-d',
       'sun-nd', 'irtp', 'vrrp', 'cbt', 'ippc', 'pvp', 'idpr', 'smp',
       'fc', 'visa', 'cftp', 'cpnx', 'encap', 'larp', 'sps', 'ipv6-route',
       'prm', 'xnet', 'cphb', 'bna', 'ipv6-opts', 'mux', 'mobile',
       'compaq-peer', 'ipv6-no', 'leaf-2', 'pim', 'wb-mon', 'scps', 'hmp',
       'ddx', 'srp', 'ipip', 'igp', 'zero', 'idrp', 'mfe-nsp', 'ipv6',
       'fire', 'rsvp', 'ggp', 'gmtp', 'skip', 'sdrp', 'ax.25', 'il',
       'sat-expak', 'wsn', 'vines', 'pipe', 'sep', 'xtp', 'rvd',
       'kryptolan', 'st2', 'trunk-1', 'crudp', 'iso-tp4', 'eigrp', 'l2tp',
       'ddp', 'argus', 'xns-idp', 'swipe', 'micp', 'ipcv', 'ipx-n-ip',
       'ifmp', 'secure-vmtp', 'rdp', 'pri-enc', 'mhrp', 'ib', 'etherip',
       'bbn-rcc', 'qnx', 'vmtp', 'iatp', 'sm', 'ipnip', 'tlsp', 'leaf-1',
       'pnni', 'sat-mon', 'merit-inp', 'nvp', 'br-sat-mon', 'uti', 'igmp',
       'ipv6-frag', 'tp++', 'narp', 'pgm', 'icmp', 'ttp']:
            self.data[f'feature2_{col}'] = self.data['feature2'].eq(col).astype(int)

        for col in ['-', 'http', 'dns', 'ftp', 'ssh', 'smtp', 'ftp-data', 'ssl',
       'pop3', 'dhcp', 'radius', 'snmp', 'irc']:
            self.data[f'feature3_{col}'] = self.data['feature3'].eq(col).astype(int)

        for col in ['FIN', 'INT', 'CON', 'REQ', 'RST', 'ECO', 'ACC']:
            self.data[f'feature4_{col}'] = self.data['feature4'].eq(col).astype(int)



            