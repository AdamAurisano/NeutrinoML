'''
Base class for input preprocessing
'''

class BaseProcessor:
    '''Base class for input graph processing'''
    def __init__(self, node_table, particle_table):
        self.node_table = node_table
        self.particle_table = particle_table
        
    def get_particle(self, idx):
        '''Return the table row corresponding to a particle index'''
        print(self.particle_table[(self.particle_table.id == idx)].squeeze().id)