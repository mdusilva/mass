import logging
import logging.handlers
import mamuto
import numpy as np
import os
import warnings

#create logger
logger = logging.getLogger(__name__)
"""show the module activity in the terminal and store debug details in a log file - **debug.log**."""
logger.setLevel(logging.DEBUG)

 # create console handler and set level to info
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# create error file handler and set level to debug
_handler = logging.handlers.RotatingFileHandler(os.path.join(os.path.dirname(__file__), "debug.log"),"w", maxBytes=1024*1024, backupCount=10, delay="true")
_handler.setLevel(logging.DEBUG)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

class Ticker(object):
    """
    Keep track of progress
    
    """

    def __init__(self, n, interval = 0.1, end=1):
        """Initialize counter"""
        self.milestones = (np.arange(0,end,interval) * n).astype("int")
        self.now = 0.
        #logger.debug("Initialized Ticker with parameters n={}, interval={} and end={}".format(str(n), str(interval), str(end)))
        #logger.debug("Milestones are: {}".format(str(self.milestones)))
    
    def tick(self, i):
        """Check if iteration advances ticker"""
        if i in self.milestones:
            mile = np.searchsorted(self.milestones, i)
            self.now = mile *10
            #logger.debug("Reached new mile, now at {}".format(str(self.now)))
            #logger.debug("Returning {}".format("True"))
            return True
        else:
            #logger.debug("Returning {}".format("False"))
            return False

    def reset(self):
        self.now = 0.
        #logger.debug("Reseting, now is {}".format(str(self.now)))    

class Trace(object):
    """
    Trace objects contain the traces of variables
    
    
    """

    def __init__(self, x0, n):
        """Trace is array with shape (x0.shape,n)"""
        self.trace = np.zeros((n,) + x0.shape)
        self.counter = 0
        #logger.debug("Initializing a Trace with paramters x0={} and n={}".format(str(x0), str(n)))
        #logger.debug("Initial trace is: {}".format(str(self.trace)))
        #logger.debug("Counter set at: {}".format(str(self.counter)))

    def add(self, x):
        """Add position to trace"""
        self.trace[self.counter] = x
        #logger.debug("Adding element to trace: {} in position {}".format(str(x), str(self.counter)))
        #logger.debug("Trace is now: {}".format(str(self.trace)))
        self.counter += 1
        #logger.debug("Counter set at: {}".format(str(self.counter)))

    def modes(self,kn=35):
        """Return the modes of the sample distributions"""
        modes = np.zeros(self.trace[0].shape)
        tree = spatial.cKDTree(self.trace)
        max_distances = []
        for element in tree.data:
            distances, indices =  tree.query(element, k=kn)
            max_distances.append(max(distances[1:]))
        mode_index = np.argmin(np.array(max_distances))
        modes = tree.data[mode_index]
        return modes

    def averages(self):
        """Return the averages of the sample distributions"""
        return np.average(self.trace, axis=0)

    def save(self, filename):
        """Save trace to file"""
        np.savetxt(filename, self.trace)

class Model(object):
    """
    Model definitions

    Create Bayesian model object defining the likelihoods and priors.
    The object includes the methodos to compute the distributions in
    parallel using the mamuto package.

    Arguments
    ---------

    p0: numpy array
        initial positions of parameters

    h0: numpy array
        initial postions of hyperparameters

    flike: function
        common likelihood of parameters
    
    fprior: function
        common prior of parameters; also the likelihood of hyperparameters

    fhyprior: function
        prior of hyperparameters

    data: numpy array
        observed data. The likelihood of parameters depends on this, in general.

    configfile: string
        file with cluster configuration that is used by mamuto package

    depends (optional): list of strings
        list of modules containing the functions, to be passed to mamuto package.
        Default is None.
    """
    
    def __init__(self, p0, h0, flike, fprior, fhyprior, data, configfile, depends=None):
        self.function_mapper = mamuto.Mapper(configfile, depends=depends)
        #logger.debug("Creating Model object with parameters p0={}, h0={}, flike={}, fprior={}, fhyprior={}, data={}".format(str(p0), str(h0), str(flike), str(fprior), str(fhyprior), str(data)))
        self.p0 = p0
        self.h0 = h0
        self.flike = flike
        self.fprior = fprior
        self.fhyprior = fhyprior
        self.data = data
        self._add_functions()

    def _add_functions(self):
        """Setup remote functions"""
        rargs = [None, self.data]
        #logger.debug("Adding functions to remote processes...")
        self.function_mapper.add_function(self.flike)
        self.function_mapper.add_function(self.fprior)
        #logger.debug("Done.")
        #logger.debug("Setting up remote parameter in remote processes: {}".format(str(rargs)))
        self.function_mapper.add_remote_arguments(self.flike, rargs)
        #logger.debug("Done.")
        
    def likelihood_terms(self, p, data=None):
        """Compute likelihood terms"""
        largs = [p, data]
        #logger.debug("Computing Likelihoods using parameters: {}".format(str(largs)))
        result = self.function_mapper.remap(self.flike, largs)
        #logger.debug("Done.")
        return result

    def prior_terms(self, p, hyper):
        """Compute prior terms"""
        largs = [p, [hyper]]
        #logger.debug("Computing Priors using parameters: {}".format(str(largs)))
        result = self.function_mapper.remap(self.fprior, largs)
        #logger.debug("Done.")
        return result

class MHSimpleUpdates(object):
    """
    Perform MH steps in 1D
    """

    def __init__(self, p0, lnlike, lnprior, sd_proposal=None):
        self._main_rand = np.random.mtrand.RandomState()
        self.sd_proposal = sd_proposal
        if sd_proposal is None:
            if p0 != 0.:
                self.sd_proposal = abs(p0)
            else:
                self.sd_proposal = 1.
        self.p = p0
        self.setprob(lnlike, lnprior)
        self.adaptive_scale_factor = 1
        #logger.debug("Created 1D Stepper with parameters p0={}, lnprob={} and sd_proposal={}".format(str(self.p), str(self.lnprob), str(self.sd_proposal)))
        self.reset()

    def reset(self):
        """Reset the stepper."""
        self.iteration = 0
        self.accepted = 0

    def step(self, q, newlnlike, newlnprior):
        """Perform a MH step or one-at-a-time steps for array variables"""
        self.iteration = self.iteration + 1
        newlnprob = newlnlike + newlnprior
        accept = np.log(self._main_rand.rand()) < newlnprob - self.lnprob
        #logger.debug("Trying new 1D step q={} with logp={}".format(str(q), str(newlnprob)))
        if accept:
            #logger.debug("Step accepted")
            self.p = q
            self.setprob(newlnlike, newlnprior)
            self.accepted = self.accepted + 1
        #logger.debug("Returning p={} with logp={}".format(str(self.p), str(self.lnprob)))
        return self.p, self.lnlike, self.lnprior

    def setprob(self, lnlike, lnprior):
        """Set the value of current point probablity: likelihood, prior and total"""
        self.lnlike = lnlike
        self.lnprior = lnprior
        self.lnprob = self.lnlike + self.lnprior

    def tune(self):
        """Tune the MH stepper, using criteria taken from PyMC"""
        acc_rate = self.accepted / self.iteration
        tuning = True
        current_factor = self.adaptive_scale_factor
        #logger.debug("Tuning proposal sd. Acceptance rate is: {}".format(str(acc_rate)))
        # Switch statement
        if acc_rate < 0.001:
            # reduce by 90 percent
            self.adaptive_scale_factor *= 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            self.adaptive_scale_factor *= 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            self.adaptive_scale_factor *= 0.9
        elif acc_rate > 0.95:
            # increase by factor of ten
            self.adaptive_scale_factor *= 10.0
        elif acc_rate > 0.75:
            # increase by double
            self.adaptive_scale_factor *= 2.0
        elif acc_rate > 0.5:
            # increase by ten percent
            self.adaptive_scale_factor *= 1.1
        else:
            tuning = False
        #logger.debug("The adaptive scale factor is now: {}".format(str(self.adaptive_scale_factor)))
        self.reset()
        # Prevent from tuning to zero
        if not self.adaptive_scale_factor:
            self.adaptive_scale_factor = current_factor
            return False
        return tuning

    def propose(self):
        """"Generate MH proposal from Gaussian distribution"""
        q = self._main_rand.normal(0, self.sd_proposal * self.adaptive_scale_factor, 1)
        q = self.p + q
        #logger.debug("Proposing new step: {}".format(str(q)))
        return q        

class MHBlockUpdates(object):
    """
    Perform MH steps in block updates
    """

    def __init__(self, p0, lnlike, lnprior, cov=None):
        self._main_rand = np.random.mtrand.RandomState()
        self.p = p0
        self.setprob(lnlike, lnprior)
        self.dim = len(self.p)
        if cov:
            self.cov_proposal = cov
        else:
            self.cov_proposal = np.eye(self.dim) * np.abs(self.p)
        self.chain_mean = np.zeros(self.dim)
        self._trace = []
        self._trace_count = 0
        #logger.debug("Created block Stepper with parameters p0={}, lnprob={} and cov={}".format(str(p0), str(self.lnprob), str(cov)))
        #logger.debug("Proposal covariance is now: {}".format(str(self.cov_proposal)))
        #logger.debug("Space dimension is: {}".format(str(self.dim)))
        self.reset()

    def reset(self):
        """Reset the stepper."""
        self.iteration = 0
        self.accepted = 0

    def setprob(self, lnlike, lnprior):
        """Set the value of current point probablity: likelihood, prior and total"""
        self.lnlike = lnlike
        self.lnprior = lnprior
        self.lnprob = self.lnlike + self.lnprior

    def step(self, q, newlnlike, newlnprior):
        """Perform a MH step in block"""
        self.iteration = self.iteration + 1
        newlnprob = newlnlike + newlnprior
        #logger.debug("Performing block update. This is iteration {}".format(str(self.iteration)))
        accept = np.log(self._main_rand.rand()) < newlnprob - self.lnprob        
        #logger.debug("Trying new block update q={} with logp={}".format(str(q), str(newlnprob)))
        if accept:
            #logger.debug("Step accepted")
            self.p = q
            self.setprob(newlnlike, newlnprior)
            self.accepted = self.accepted + 1
        self._trace.append(self.p)
        #logger.debug("Returning p={} with logp={}".format(str(self.p), str(self.lnprob)))
        return self.p, self.lnlike, self.lnprior

    def adapt_covariance(self):
        """Adapt covariance matrix"""
        scaling = (2.4) ** 2 / self.dim # Gelman et al. 1996.
        epsilon = 1.0e-5
        chain = np.asarray(self._trace)
        previous_cov = self.cov_proposal
        # Recursively compute the chain mean
        #logger.debug("Adapting covariance matrix. Covariance is now: {}".format(str(self.cov_proposal)))
        self.cov_proposal, self.chain_mean = self.recursive_cov(self.cov_proposal,  self._trace_count, self.chain_mean, chain, scaling=scaling, epsilon=epsilon)
        #logger.debug("New covariance is: {}".format(str(self.cov_proposal)))
        #logger.debug("Chain mean is: {}".format(str(self.chain_mean)))
        self.shrink()
        #try:
        #    self._check_cov()
        #except np.linalg.LinAlgError:
        #    warnings.warn("Covariance matrix is not positive definite." + "\n" + "Keeping previous value.")
        #    self.cov_proposal = previous_cov
        self._trace_count += len(self._trace)
        self._trace = []

    def _check_cov(self):
        """Check if covariance matrix is positive definite"""
        c = np.linalg.cholesky(self.cov_proposal)

    def recursive_cov(self, cov, length, mean, chain, scaling=1, epsilon=0):
        r"""Compute the covariance recursively.
        Return the new covariance and the new mean.
        .. math::
            C_k & = \frac{1}{k-1} (\sum_{i=1}^k x_i x_i^T - k\bar{x_k}\bar{x_k}^T)
            C_n & = \frac{1}{n-1} (\sum_{i=1}^k x_i x_i^T + \sum_{i=k+1}^n x_i x_i^T - n\bar{x_n}\bar{x_n}^T)
                & = \frac{1}{n-1} ((k-1)C_k + k\bar{x_k}\bar{x_k}^T + \sum_{i=k+1}^n x_i x_i^T - n\bar{x_n}\bar{x_n}^T)
        :Parameters:
            -  cov : matrix
                Previous covariance matrix.
            -  length : int
                Length of chain used to compute the previous covariance.
            -  mean : array
                Previous mean.
            -  chain : array
                Sample used to update covariance.
            -  scaling : float
                Scaling parameter
            -  epsilon : float
                Set to a small value to avoid singular matrices.
        """
        n = length + len(chain)
        k = length
        new_mean = self.recursive_mean(mean, length, chain)

        t0 = k * np.outer(mean, mean)
        t1 = np.dot(chain.T, chain)
        t2 = n * np.outer(new_mean, new_mean)
        t3 = epsilon * np.eye(cov.shape[0])

        new_cov = (
            k - 1) / (
                n - 1.) * cov + scaling / (
                    n - 1.) * (
                        t0 + t1 - t2 + t3)
        return new_cov, new_mean

    def recursive_mean(self, mean, length, chain):
        r"""Compute the chain mean recursively.
        Instead of computing the mean :math:`\bar{x_n}` of the entire chain,
        use the last computed mean :math:`bar{x_j}` and the tail of the chain
        to recursively estimate the mean.
        .. math::
            \bar{x_n} & = \frac{1}{n} \sum_{i=1}^n x_i
                      & = \frac{1}{n} (\sum_{i=1}^j x_i + \sum_{i=j+1}^n x_i)
                      & = \frac{j\bar{x_j}}{n} + \frac{\sum_{i=j+1}^n x_i}{n}
        :Parameters:
            -  mean : array
                Previous mean.
            -  length : int
                Length of chain used to compute the previous mean.
            -  chain : array
                Sample used to update mean.
        """
        n = length + len(chain)
        return length * mean / n + chain.sum(0) / n        

    def shrink(self):
        """Shrink covariance matrix if acceptance rate is too small (taken from PyMC)"""
        acc_rate = self.accepted / self.iteration
        tuning = True
        #logger.debug("Shrinking covaraince matrix. Acceptance rate is {}".format(str(acc_rate)))
        if acc_rate < .001:
            self.cov_proposal *= .01
        elif acc_rate < .01:
            self.cov_proposal *= .25
        else:
            tuning = False
        #logger.debug("Covariance is now: {}".format(str(self.cov_proposal)))
        self.reset()
        return tuning

    def tune(self):
        """Tuning done internally only, not called"""
        return False


    def propose(self):
        """Generate MH proposal from Gaussian distribution."""
        q = self._main_rand.multivariate_normal(np.zeros(self.p.shape), self.cov_proposal, 1)
        #logger.debug("Proposing new step: {}".format(str(q)))
        q = self.p + q[0]
        return q

class Mcmc(object):
    """
    MCMC sampler

    Perform MCMC sampling given a Bayesian model
    """

    def __init__(self, model):
        self.model = model
        #logger.debug("MCMC sampler created.")
        self._main_rand = np.random.mtrand.RandomState()

    def sample(self, n, p0, h0, thin=1, burnin=0, sampling_method="block", temperatures=[1], tempering=False, cov_proposal=None, adaptcov=False, tune_throughout=True, tune_interval=1000, adaptation_delay=1000, adaptation_interval=200):
        """
        Sample Markov Chain

        Fit the probability model using MCMC.

        Arguments
        ---------

        n: integer
            number of iterations (except burnin)

        p0: numpy array
            initial positions of the parameters. Dimension is (nb*nw, np), the number of lines is the number of blocks of variables (nb) times
            the number of walkers (nw) used for each block; the number of columns is the number of parameters per bock (np)

        h0: numpy array
            initial positions of the hyperparameters.

        thin (optional): integer
            thinning parameter: the chain states is only saved every `thin` steps. Default is 1.

        burnin (optional): integer
            burn-in period: this iterations are done in the beggining of the sampling and then discarded. Default is 0.

        sampling_method: "cycle" or "block" string
            which sampling method to use. Cycle through the parameters, one-at-a-time or perform block updates.

        temperatures (optional): list of floats between 1 and 0
            temperatures to use for parallel tempering. The number of temperatures defines the number of walkers, so must
            be consistent with p0. Default is [1].

        tempering: boolean
            whether to perform parallel tempering

        cov_proposal (optional): numpy array
            Covariance matrix of the proposal function when using "block" sampling method. Dimension is (np, np) where np
            is the number of parameters. If using the adaptive covariance option this is only the initial value.
            Default is None, which means it will be estimated from data.

        adaptcov: boolean
            whether to adapt the covariance matrix when using "block" sampling method

        tune_throughout: boolean
            whether to tune the the proposal function after the burnin period, when using the "cycle" sampling
            method

        tune_interval: integer
            interval between successive tuning attempts (used by the "cycle" sampling method). Default is 1000.

        adaptation_delay: integer
            number of iterations before any attempt to tune the proposal covariance mtrix (used by the "block"
            sampling method). Default is 1000.

        adaptation_interval: integer
            number of iterations between successive adaptions of the proposal covariance matrix (used by
             the "block" sampling method). Default is 200
        
        """
        self.thin = int(thin)
        self.burnin = int(burnin)
        self.iterations = int(n) + self.burnin
        self.tune_interval = int(tune_interval)
        self.adaptation_delay = int(adaptation_delay)
        self.adaptation_interval = int(adaptation_interval)
        self.cov_proposal = cov_proposal
        self.adaptcov = adaptcov

        self.sampling_method = sampling_method
        self.temperatures = np.array(temperatures)
        self._check_temp()
        self.nwalkers = len(self.temperatures)
        self.nchains = len(p0)
        self.nblocks = self.nchains // self.nwalkers
        self.blockidx = np.arange(0,  self.nchains, self.nwalkers)
        self._temp_array = np.tile(self.temperatures, self.nblocks)
        self.setcup(p0, h0)

        logger.info("Starting sampling using method {}.".format(self.sampling_method))
        logger.info("Doing {} iterations with a burnin period of {}, thining by {}.".format(str(self.iterations), str(self.burnin), str(self.thin)))
        logger.info("Using {} walkers with {} temperatures.".format(str(self.nwalkers), str(self.temperatures)))
        #logger.debug("The complete array of temperatures is: {}".format(str(self._temp_array)))
        logger.info("There are {} blocks of variables, so running {} chains in total.".format(str(self.nblocks), str(self.nchains)))
        #logger.debug("Initial values for the chains are: {}".format(str(self.p)))
        #logger.debug("Inital value for the hyperparameter is {}".format(str(self.hyper)))
        #logger.debug("The indices of the blocks are: {}".format(str(self.blockidx)))
        #logger.debug("Obtaining dimensions of the two parameter spaces...")

        self.getdim()

        #logger.debug("X dim is {} and Hyper dim is {}".format(str(self.ndim), str(self.hdim)))

        #logger.debug("Initializing walkers (chains)...")
        
        self._initwalkers(self.p, self.hyper)
        self.tempering = tempering
        self.tune_throughout = tune_throughout
        if self.tempering:
            if self.nwalkers == 1:
                logger.info("Parallel tempering requires more than one temperature, setting to False.")
                self.tempering = False

        self._inittraces(int((self.iterations - self.burnin)/thin), self.p, self.hyper)

        self.progress = Ticker(self.iterations)

        for n_iteration in range(self.iterations):
            #logger.debug("Current iteration: {}".format(str(n_iteration)))
            if self.sampling_method == "cycle":
                self.histep()
                self.cyclestep()
                if self.tempering:
                    self.p = self.coupling_update(self.p)
                #self.priors = self.model.prior_terms(self.p, self.hyper)
                #self.histep()
                if n_iteration < self.burnin or self.tune_throughout:
                    if n_iteration % tune_interval == 0:
                        self.cycletune()
                if n_iteration >= self.burnin and n_iteration % thin == 0:
                    p_blocks = self.p[self.blockidx]
                    p_blocks = self.p.ravel()
                    for trace, p in zip(self.traces, p_blocks):
                        trace.add(p)
                    for trace, h in zip(self.hypertraces, self.hyper):
                        trace.add(h)
            
            if self.sampling_method == "block":
                self.histep()
                self.blockstep()
                if self.tempering:
                    self.p = self.coupling_update(self.p)
                #self.priors = self.model.prior_terms(self.p, self.hyper)
                #self.histep()
                if self.adaptcov:
                    if n_iteration > self.adaptation_delay and n_iteration % self.adaptation_interval == 0:
                        self.blocktune()
                if n_iteration >= self.burnin and n_iteration % thin == 0:
                    p_blocks = self.p[self.blockidx]
                    for trace, p in zip(self.traces, p_blocks):
                        trace.add(p)
                    self.hypertraces[0].add(self.hyper)
            if self.progress.tick(n_iteration):
                logger.info("Progress: {}%".format(str(self.progress.now)))
        logger.info("Progress: {}%".format(str(100)))

    def getdim(self):
        """Compute dimensions of the parameter spaces (X and Hyper)."""
        if isinstance(self.p[0], np.ndarray):
            self.ndim = len(self.p[0])
        else:
            self.ndim = 1
        if isinstance(self.hyper, np.ndarray):
            self.hdim = len(self.hyper)
        else:
            self.hdim = 1
    
    def _check_temp(self):
        """Check temperature array for problems"""
        if  self.temperatures[0] != 1.:
            self.temperatures[0] = 1.
            logger.warning("First temperature must be = 1. Changed to the correct value.")

    def _inittraces(self, n_iterations, p, hyper):
        """Initalize traces of all variables"""
        p_blocks = p[self.blockidx]
        if  self.sampling_method == "cycle":
            self.traces = [Trace(p_blocks[i,j],n_iterations) for i in range(self.nblocks) for j in range(self.ndim)]
            self.hypertraces = [Trace(hyper[i], n_iterations) for i in range(self.hdim)]
        elif self.sampling_method == "block":
            self.traces = [Trace(p_blocks[i], n_iterations) for i in range(self.nblocks)]
            self.hypertraces = [Trace(hyper, n_iterations)]


    def _initwalkers(self, p, hyper):
        """Initialize walkers"""
        #logger.debug("Computing likelihoods with p = {}...".format(str(p)))
        likes = self.model.likelihood_terms(p)
        #logger.debug("Likelihoods = {}".format(str(likes)))
        #logger.debug("Computing priors with p = {} and hyper = {} ...".format(str(p), str(hyper)))
        priors = self.model.prior_terms(p, hyper)
        #logger.debug("Priors = {}".format(str(priors)))
        self.likes = np.array(likes) * self._temp_array
        self.priors = np.array(priors) * self._temp_array
        #logger.debug("Computing initial value of the Hyperprior...")
        self.probhyp = self.model.fhyprior(self.hyper)
        #self.hyper_accepted = 0
        #logger.debug("Hyperprior = {}".format(str(self.probhyp)))
        #self.logprobs = np.array(likes) + np.array(priors)
        #logger.debug("Total log prob = {}".format(str(self.likes+self.priors)))
        #self.logprobs = self.logprobs * self._temp_array
        #logger.debug("Total log probs tempered = {}".format(str(self.likes+self.priors)))
        if  self.sampling_method == "cycle":
            self.walkers = [[MHSimpleUpdates(p[i,j], self.likes[i], self.priors[i]) for j in range(self.ndim)] for i in range(self.nchains)]
            self.hyperwalkers = [MHSimpleUpdates(hyper[i], np.sum(self.priors[self.blockidx]), self.probhyp) for i in range(self.hdim)]
            #logger.debug("Initialized list of walkers with {} elements".format(str(len(self.walkers))))
            #logger.debug("Initialized list of hyper walkers with {} elements".format(str(len(self.hyperwalkers))))
            #self.walkers = [[MHSimpleUpdates(self.p[i,j], self.logprobs[i]) for i in range(self.nchains)] for j in range(self.ndim)]
        elif self.sampling_method == "block":
            self.walkers = [MHBlockUpdates(v_p, self.likes[i_p], self.priors[i_p], cov=self.cov_proposal) for v_p, i_p in zip(p, range(self.nchains))]
            self.hyperwalkers = MHBlockUpdates(hyper, np.sum(self.priors[self.blockidx]), self.probhyp)
            #logger.debug("Initialized list of walkers with {} elements".format(str(len(self.walkers))))
            #logger.debug("Initialized list of hyper walkers with 1 element")

    def setcup(self, p, hyper):
        """Set current position in parameter space. Arrays are converted to float dtype"""
        #self.p = p.astype(float)
        new_p = np.atleast_2d(p.astype(float))
        if new_p.shape[0] == 1:
            self.p = new_p.T
        else:
            self.p = new_p
        #self.hyper = np.atleast_2d(hyper)
        self.hyper = hyper.astype(float)

    def cycletune(self):
        """Tune all walkers when using 1D updates"""
        #logger.debug("Tuning walkers for cycle sampling...")
        tuning = [self.walkers[i][j].tune() for j in range(self.ndim) for i in range(self.nchains)]
        #logger.debug("Tuning is: {}".format(str(tuning)))
        htuning = [self.hyperwalkers[i].tune() for i in range(self.hdim)]
        #logger.debug("Tuning of hyperwalker is: {}".format(str(htuning)))
        return tuning, htuning

    def blocktune(self):
        """Tune all walkers when using block updates"""
        #logger.debug("Tuning walkers for block sampling...")
        tuning = [self.walkers[i].adapt_covariance() for i in range(self.nchains)]
        #logger.debug("Tuning is: {}".format(str(tuning)))
        htuning = self.hyperwalkers.adapt_covariance()
        #logger.debug("Tuning of hyperwalker is: {}".format(str(htuning)))
        return tuning, htuning

    def cyclestep(self):
        """Perform a single step using cycling 1D updates"""
        q = self.p
        #logger.debug("Performing {} total steps in 1D".format(str(self.nchains * self.ndim)))
        #logger.debug("Current positions are: {}".format(str(q)))
        for i in range(self.ndim):
            #logger.debug("Updating dimension: {}".format(str(i)))
            #update probability values for all walkers
            #logger.debug("Current likelihoods are: {}".format(str(self.likes)))
            #logger.debug("Current priors are: {}".format(str(self.priors)))
            [self.walkers[j][i].setprob(self.likes[j], self.priors[j]) for j in range(self.nchains)]
            #logger.debug("Proposing values for dim {}".format(str(i)))
            q_walker = [w[i].propose()[0] for w in self.walkers]
            q_walker = np.array(q_walker)
            #logger.debug("Proposed values are: {}".format(str(q_walker)))
            q[:,i] = q_walker
            #logger.debug("Computing likelihoods with p = {}...".format(str(q)))
            newlikes = self.model.likelihood_terms(q)
            #logger.debug("Likelihoods = {}".format(str(newlikes)))
            #logger.debug("Computing priors with p = {} and hyper = {} ...".format(str(q), str(self.hyper)))
            newpriors = self.model.prior_terms(q, self.hyper)
            #logger.debug("Priors = {}".format(str(newpriors)))
            new_positions = [self.walkers[j][i].step(q[j,i], self._temp_array[j] * newlikes[j], self._temp_array[j] * newpriors[j]) for j in range(self.nchains)]
            new_p = [x[0] for x in new_positions]
            new_1 = [x[1] for x in new_positions]
            new_2 = [x[2] for x in new_positions]
            self.likes = np.array(new_1)
            self.priors = np.array(new_2)
            #logger.debug("New positions are: {}".format(str(new_p)))
            #logger.debug("Current likelihoods are: {}".format(str(self.likes)))
            #logger.debug("Current priors are: {}".format(str(self.priors)))
            q[:,i] = np.array(new_p)
        #logger.debug("Current total prob is: {}".format(str(self.likes+self.priors)))
        #logger.debug("New position after updating all dimensions: {}".format(str(q)))
        self.p = q


    def blockstep(self):
        """Perform a single step using block updates"""
        #logger.debug("Performing {} total steps in {}D".format(str(self.nchains), str(self.ndim)))
        #logger.debug("Current positions are: {}".format(str(self.p)))
        #update probability values for all walkers
        #logger.debug("Current likelihoods are: {}".format(str(self.likes)))
        #logger.debug("Current priors are: {}".format(str(self.priors)))
        [self.walkers[i].setprob(self.likes[i], self.priors[i]) for i in range(self.nchains)]
        q = [w.propose() for w in self.walkers]
        q = np.array(q)
        #logger.debug("Proposed positons are: {}".format(str(q)))
        #logger.debug("Computing likelihoods with p = {}...".format(str(q)))
        newlikes = self.model.likelihood_terms(q)
        #logger.debug("Likelihoods = {}".format(str(newlikes)))
        #logger.debug("Computing priors with p = {} and hyper = {} ...".format(str(q), str(self.hyper)))
        newpriors = self.model.prior_terms(q, self.hyper)
        #logger.debug("Priors = {}".format(str(newpriors)))
        new_positions = [self.walkers[j].step(q[j], self._temp_array[j] * newlikes[j], self._temp_array[j] * newpriors[j]) for j in range(self.nchains)]
        new_p = [x[0] for x in new_positions]
        new_1 = [x[1] for x in new_positions]
        new_2 = [x[2] for x in new_positions]
        self.likes = np.array(new_1)
        self.priors = np.array(new_2)
        q = np.array(new_p)
        #logger.debug("New positions are: {}".format(str(q)))
        #logger.debug("Current likelihoods are: {}".format(str(self.likes)))
        #logger.debug("Current priors are: {}".format(str(self.priors)))
        #logger.debug("Current total prob is: {}".format(str(self.likes+self.priors)))
        self.p = q

    def histep(self):
        """Perform a step in the hyperparameter."""
        phyp = self.hyper
        #logger.debug("Performing step in the hyperparameter. Initial position: {}".format(str(phyp)))
        if  self.sampling_method == "cycle":           
            q = phyp
            qhyp = [w.propose() for w in self.hyperwalkers]
            qhyp = np.array(qhyp)
            #logger.debug("Proposed positions are: {}".format(str(qhyp)))
            for i in range(self.hdim):
                #update prior values for all hyperwalkers
                #logger.debug("Current priors are: {}".format(str(self.priors)))
                #logger.debug("Summing to: {}".format(str(sum(self.priors))))
                #logger.debug("Current effective priors are: {}".format(str(self.priors[self.blockidx])))
                #logger.debug("Summing to: {}".format(str(sum(self.priors[self.blockidx]))))                
                #logger.debug("Current hyperprior is: {}".format(str(self.probhyp)))
                self.hyperwalkers[i].setprob(np.sum(self.priors[self.blockidx]), self.probhyp)
                q[i] = qhyp[i]
                #logger.debug("Computing priors with p = {} and hyper = {} ...".format(str(self.p), str(q)))
                newh = self.model.prior_terms(self.p, q)
                newh = np.array(newh)
                #logger.debug("Priors = {}".format(str(newh)))
                #logger.debug("Computing hyperprior...")
                newp = self.model.fhyprior(qhyp)
                #logger.debug("Hyperprior = {}".format(str(newp)))
                #Save number of accepted moves to check if step is accepted
                naccepted = self.hyperwalkers[i].accepted
                #newh = newh * self._temp_array[self.blockidx]
                #logger.debug("Tempered Priors = {}".format(str(sum(newh * self._temp_array))))
                #logger.debug(" Effective tempered Priors = {}".format(str(sum(newh[self.blockidx] * self._temp_array[self.blockidx]))))
                new_position = self.hyperwalkers[i].step(q[i], np.sum(newh[self.blockidx] * self._temp_array[self.blockidx]), newp)
                #logger.debug("New position: {}".format(str(new_position[0])))
                #Save new priors if step was accepted
                if self.hyperwalkers[i].accepted > naccepted:
                    self.priors = newh * self._temp_array
                    self.probhyp = newp
                phyp[i] = new_position[0]
                #logger.debug("Current priors are: {}".format(str(self.priors)))
                #logger.debug("Summing to: {}".format(str(sum(self.priors))))
                #logger.debug("Current effective priors are: {}".format(str(self.priors[self.blockidx])))
                #logger.debug("Summing to: {}".format(str(sum(self.priors[self.blockidx])))) 
                #logger.debug("Current hyperprior is: {}".format(str(self.probhyp)))

        elif self.sampling_method == "block":
            #update prior values for all hyperwalkers
            #logger.debug("Current priors are: {}".format(str(self.priors)))
            #logger.debug("Summing to: {}".format(str(sum(self.priors))))
            #logger.debug("Current effective priors are: {}".format(str(self.priors[self.blockidx])))
            #logger.debug("Summing to: {}".format(str(sum(self.priors[self.blockidx])))) 
            #logger.debug("Current hyperprior is: {}".format(str(self.probhyp)))
            self.hyperwalkers.setprob(np.sum(self.priors[self.blockidx]), self.probhyp)
            qhyp = self.hyperwalkers.propose()
            #logger.debug("Proposed positions are: {}".format(str(qhyp)))
            #logger.debug("Computing priors with p = {} and hyper = {} ...".format(str(self.p), str(qhyp)))
            newh = self.model.prior_terms(self.p, qhyp)
            newh = np.array(newh)
            #logger.debug("Priors = {}".format(str(newh)))
            #logger.debug("Computing hyperprior...")
            newp = self.model.fhyprior(qhyp)
            #logger.debug("Hyperprior = {}".format(str(newp)))
            #Save number of accepted moves to check if step is accepted
            naccepted = self.hyperwalkers.accepted
            #logger.debug("Tempered Priors = {}".format(str(sum(newh * self._temp_array))))
            #logger.debug(" Effective tempered Priors = {}".format(str(sum(newh[self.blockidx] * self._temp_array[self.blockidx]))))
            new_positions = self.hyperwalkers.step(qhyp, np.sum(newh[self.blockidx] * self._temp_array[self.blockidx]), newp )
            if self.hyperwalkers.accepted > naccepted:
                self.priors[self.blockidx] = newh * self._temp_array
                self.probhyp = newp
            phyp = new_positions[0]
            #logger.debug("Current priors are: {}".format(str(self.priors)))
            #logger.debug("Summing to: {}".format(str(sum(self.priors))))
            #logger.debug("Current effective priors are: {}".format(str(self.priors[self.blockidx])))
            #logger.debug("Summing to: {}".format(str(sum(self.priors[self.blockidx])))) 
            #logger.debug("Current hyperprior is: {}".format(str(self.probhyp)))
        phyp = np.array(phyp)
        #logger.debug("New hyperparameter is: {}".format(str(phyp)))
        self.hyper = phyp    

    def coupling_update(self, p):
        """Parallel tempering coupling update"""
        #data = self.model.data
        #logger.debug("Performing the coupling update...")
        #logger.debug("Inital p = {}".format(str(p)))
        #logger.debug("Inital Likes = {}".format(str(self.likes)))
        #logger.debug("Initial priors = {}".format(str(self.priors)))
        #logger.debug("Inital total prob = {}".format(str(self.likes+self.priors)))
        s = np.array([np.random.choice(self.nwalkers, 2, replace=False) for i in range(self.nblocks)])
        #logger.debug("Proposed swaps: {}".format(str(s)))
        #idx = self.blockidx
        #p1 = p[s[:,0]+idx]
        #p2 = p[s[:,1]+idx]
        #data1 = data[s[:,1]+idx]
        #data2 = data[s[:,0]+idx]
        idx = [s[i] + self.blockidx[i] for i in range(self.nblocks)]
        idx = np.array(idx).ravel()
        idx0 = [s[i][::-1] + self.blockidx[i] for i in range(self.nblocks)]
        idx0 = np.array(idx0).ravel()
        #data_idx = [np.array([0,1]) + self.blockidx[i] for i in range(self.nblocks)]
        #data_idx = np.array(data_idx).ravel()
        #logger.debug("Corrected indices: {}".format(str(idx)))
        #pjob = np.concatenate((p1, p2))
        pjob = np.copy(p)
        pjob[idx0] = pjob[idx]
        #datajob = data[data_idx]
        #logger.debug("Computing likelihoods with p = {}...".format(str(pjob)))
        newlikes = self.model.likelihood_terms(pjob)
        newlikes = np.array(newlikes)
        #logger.debug("Likelihoods = {}".format(str(newlikes)))
        #logger.debug("Computing priors with p = {} and hyper = {} ...".format(str(pjob), str(self.hyper)))
        newpriors = self.model.prior_terms(pjob, self.hyper)
        newpriors = np.array(newpriors)
        #logger.debug("Priors = {}".format(str(newpriors)))
        #f1 = newlikes[0:self.nblocks] + newpriors[0:self.nblocks]
        #f2 = newlikes[self.nblocks:] + newpriors[self.nblocks:]
        #f1 = newlikes[0::2] + newpriors[0::2]
        #f2 = newlikes[1::2] + newpriors[1::2]
        f1 = newlikes[s[:,0] + self.blockidx] + newpriors[s[:,0] + self.blockidx]
        f2 = newlikes[s[:,1] + self.blockidx] + newpriors[s[:,1] + self.blockidx]
        #logger.debug("Total probs = {}".format(str(newlikes+newpriors)))
        #logger.debug("F1: {} and F2: {}".format(str(f1), str(f2)))
        #t1 = self._temp_array[s[:,0]+idx]
        #t2 = self._temp_array[s[:,1]+idx]
        _temp_temp_array = self._temp_array[idx]
        t1 = _temp_temp_array[0::2]
        t2 = _temp_temp_array[1::2]
        #logger.debug("Temperatures: {} and {}".format(str(t1), str(t2)))
        #logA = [f2[i] * t1[i] + f1[i] * t2[i] - f1[i] * t1[i] - f2[i] * t2[i] for i in range(self.nblocks)]
        idx = self.blockidx
        #logger.debug("Positions before cycle: {}".format(str(p)))
        for i in range(self.nblocks):
            #logger.debug("Updating block: {}".format(str(i)))
            logA = f2[i] * t1[i] + f1[i] * t2[i] - f1[i] * t1[i] - f2[i] * t2[i]
            #logger.debug("Log A = {}".format(str(logA)))
            #logger.debug("F1 value = {}".format(str(f1[i])))
            #logger.debug("F2 value = {}".format(str(f2[i])))
            #logger.debug("T1 value = {}".format(str(t1[i])))
            #logger.debug("T2 value = {}".format(str(t2[i])))
            #logger.debug("Currentm indices: {} and {}".format(str(s[i,0]+idx[i]), str(s[i,1]+idx[i])))
            #logger.debug("Old likes: {} and {}".format(str(self.likes[s[i,0]+idx[i]]), str(self.likes[s[i,1]+idx[i]])))
            #logger.debug("Old priors: {} and {}".format(str(self.priors[s[i,0]+idx[i]]), str(self.priors[s[i,1]+idx[i]])))
            #logger.debug("Old total probs: {} and {}".format(str(self.priors[s[i,0]+idx[i]] + self.likes[s[i,0]+idx[i]]), str(self.priors[s[i,1]+idx[i]] + self.likes[s[i,1]+idx[i]])))
            if np.log(self._main_rand.rand()) < logA:
                #logger.debug("Position {} before switch: {}".format(str(i), str(p[s[i]+idx[i]])))
                p[s[i]+idx[i]] = np.flipud(p[s[i]+idx[i]])
                #logger.debug("Position {} after switch: {}".format(str(i), str(p[s[i]+idx[i]])))
                #logger.debug("Updating probabilites...")
                self.likes[s[i]+idx[i]] = newlikes[s[i]+idx[i]] * self._temp_array[s[i]+idx[i]]
                self.priors[s[i]+idx[i]] = newpriors[s[i]+idx[i]] * self._temp_array[s[i]+idx[i]]
                if  self.sampling_method == "cycle": 
                    for d in range(self.ndim):
                        #logger.debug("Updating dimension {} of walker {} with {}".format(str(d), str(s[i,0]+idx[i]), str(p[s[i,0]+idx[i], d])))
                        #logger.debug("Updating dimension {} of walker {} with {}".format(str(d), str(s[i,1]+idx[i]), str(p[s[i,1]+idx[i], d])))
                        #logger.debug("Old postions: {} and {}".format(str(self.walkers[s[i,0]+idx[i]][d].p), str(self.walkers[s[i,1]+idx[i]][d].p)))
                        self.walkers[s[i,0]+idx[i]][d].p =  p[s[i,0]+idx[i], d]
                        self.walkers[s[i,1]+idx[i]][d].p =  p[s[i,1]+idx[i], d]
                        #logger.debug("New postions: {} and {}".format(str(self.walkers[s[i,0]+idx[i]][d].p), str(self.walkers[s[i,1]+idx[i]][d].p)))
        #                 #logger.debug("Old probs: {} and {}".format(str(self.walkers[s[i,0]+idx[i]][d].lnprob), str(self.walkers[s[i,1]+idx[i]][d].lnprob)))
        #                 #self.walkers[s[i,0]][d].lnprob = f2[i] * t1[i]
        #                 #self.walkers[s[i,1]][d].lnprob = f1[i] * t2[i]
        #                 #logger.debug("Indices: {} and {}".format(str(idx[i]), str(idx[i]+1)))
        #                 self.walkers[s[i,0]+idx[i]][d].setprob(newlikes[idx[i]+1] * t1[i], newpriors[idx[i]+1] * t1[i])
        #                 self.walkers[s[i,1]+idx[i]][d].setprob(newlikes[idx[i]] * t2[i], newpriors[idx[i]] * t2[i])
        #                 #logger.debug("New probs: {} and {}".format(str(self.walkers[s[i,0]+idx[i]][d].lnprob), str(self.walkers[s[i,1]+idx[i]][d].lnprob)))
        #             #logger.debug("Current total probs: {}".format(str([self.walkers[i][j].lnprob for i in range(self.nchains) for j in range(self.ndim)])))
                elif self.sampling_method == "block":
                    #logger.debug("Updating walker {} with {}".format(str(s[i,0]+idx[i]), str(p[s[i,0]+idx[i]])))
                    #logger.debug("Updating walker {} with {}".format(str(s[i,1]+idx[i]), str(p[s[i,1]+idx[i]])))
                    #logger.debug("Old postions: {} and {}".format(str( self.walkers[s[i,0]+idx[i]].p), str( self.walkers[s[i,1]+idx[i]].p)))
                    self.walkers[s[i,0]+idx[i]].p =  p[s[i,0]+idx[i]]
                    self.walkers[s[i,1]+idx[i]].p =  p[s[i,1]+idx[i]]
                    #logger.debug("New positions: {} and {}".format(str( self.walkers[s[i,0]+idx[i]].p), str( self.walkers[s[i,1]+idx[i]].p)))
        #             #logger.debug("Old probs: {} and {}".format(str(self.walkers[s[i,0]+idx[i]].lnprob), str(self.walkers[s[i,1]+idx[i]].lnprob)))
        #             #self.walkers[s[i,0]].lnprob = f2[i] * t1[i]
        #             #self.walkers[s[i,1]].lnprob = f1[i] * t2[i]
        #             self.walkers[s[i,0]+idx[i]].setprob(newlikes[idx[i]+1] * t1[i], newpriors[idx[i]+1] * t1[i])
        #             self.walkers[s[i,1]+idx[i]].setprob(newlikes[idx[i]] * t2[i], newpriors[idx[i]] * t2[i])
        #             #logger.debug("New probs: {} and {}".format(str(self.walkers[s[i,0]+idx[i]].lnprob), str(self.walkers[s[i,1]+idx[i]].lnprob)))
        #logger.debug("Total new positions: {}".format(str(p)))
        #logger.debug("Total new probs: {}".format(str(self.likes+self.priors)))
        return p
