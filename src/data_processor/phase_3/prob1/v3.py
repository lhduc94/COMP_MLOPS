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
        self.one_hot_1()
        self.convert_to_categorical()
        out = self.data.copy()
        self.flush()
        return out[self.FEATURES]
    def one_hot_1(self):
        for val in ['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'pop3', 'ssh', 'dhcp',
       'ssl', 'snmp', 'irc', 'radius']:
            col_name = f'feature3_{val}'
            self.data[col_name] = self.data['feature3'].eq(val)
            if col_name not in self.FEATURES:
                self.FEATURES.append(col_name)
        
        for val in ['FIN', 'INT', 'CON', 'REQ', 'RST', 'ECO', 'ACC']:
            col_name = f'feature4_{val}'
            self.data[col_name] = self.data['feature4'].eq(val)
            if col_name not in self.FEATURES:
                self.FEATURES.append(col_name)
        
        for val in ['tcp', 'udp', 'unas', 'arp', 'ospf', 'sctp', 'any', 'gre', 'ipv6',
       'pim', 'swipe', 'mobile', 'sep', 'rsvp', 'sun-nd', 'tcf', 'mtp', 'pipe',
       'emcon', 'narp', 'ip', 'sat-expak', 'visa', 'vines', 'sprite-rpc',
       'trunk-2', 'ifmp', 'ipv6-opts', 'ipcomp', 'ptp', 'fc', 'iso-ip', 'ipcv',
       'micp', 'wsn', 'etherip', 'cbt', 'sm', 'eigrp', 'sat-mon', 'pri-enc',
       'stp', 'qnx', 'bbn-rcc', 'ddx', 'sccopmce', 'ipv6-no', 'ipv6-route',
       'egp', 'rvd', 'kryptolan', 'l2tp', 'skip', 'ipnip', 'hmp', 'dcn', 'smp',
       'crtp', 'encap', 'pup', 'compaq-peer', 'aris', 'a/n', 'scps',
       'merit-inp', 'ipx-n-ip', 'zero', 'pnni', 'nsfnet-igp', 'chaos',
       'wb-mon', 'larp', 'pgm', 'prm', 'xnet', 'irtp', 'i-nlsp', 'dgp', 'iatp',
       'bna', 'secure-vmtp', 'xns-idp', 'rdp', 'br-sat-mon', 'ib', 'ipv6-frag',
       'cpnx', 'iso-tp4', 'ddp', 'trunk-1', 'leaf-1', 'iplt', 'xtp', 'il',
       'ggp', 'uti', 'igp', 'srp', 'mux', 'st2','pvp', 'tlsp', 'gmtp', 'idpr-cmtp', 'crudp', 'wb-expak', 'ttp', 'fire',
       'leaf-2', 'mfe-nsp', 'cftp', 'ippc', 'snp', 'argus', 'vmtp', 'ax.25',
       'idpr', 'ipip', 'cphb', '3pc', 'vrrp', 'sdrp', 'nvp', 'aes-sp3-d',
       'sps', 'mhrp', 'tp++', 'idrp', 'isis', 'netblt', 'igmp', 'icmp']:
            col_name = f'feature2_{val}'
            self.data[col_name] = self.data['feature2'].eq(val)
            if col_name not in self.FEATURES:
                self.FEATURES.append(col_name)