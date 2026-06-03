import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F #added on 2024/2/1

class SemiD2Model(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        # parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        # if is_train:
        #     parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        #     parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        #     parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['LD_A', 'GD_A','G_LA', 'G_GA', 'cycle_A', 'idt_A', 'LD_B', 'GD_B', 'G_LB', 'G_GB', 'cycle_B', 'idt_B',
                           'LD_A_a','GD_A_a','G_LA_a','G_GA_a', 'LD_B_a','GD_B_a','G_LB_a','G_GB_a', 'G_A_L1', 'G_B_L1']
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B',
        #                    'G_A_a', 'G_B_a', 'G_A_L1', 'G_B_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'real_A_a', 'fake_B_a']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'real_B_a', 'fake_A_a']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'LD_A', 'LD_B', 'GD_A', 'GD_B', 'LD_A_a', 'LD_B_a', 'GD_A_a', 'GD_B_a',]
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netLD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netLD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netLD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netLD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netGD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netGD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netLD_A_a = networks.define_D(opt.output_nc, opt.ndf, opt.netLD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netLD_B_a = networks.define_D(opt.input_nc, opt.ndf, opt.netLD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGD_A_a = networks.define_D(opt.output_nc, opt.ndf, opt.netGD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGD_B_a = networks.define_D(opt.input_nc, opt.ndf, opt.netGD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_pool_a = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool_a = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # self.criterionCycle = torch.nn.L1Loss()
            # self.criterionIdt = torch.nn.L1Loss()
            # self.criterionL1 = torch.nn.L1Loss() # For pix2pix 21/05/11
            self.criterionCycle = F.smooth_l1_loss#
            self.criterionIdt = F.smooth_l1_loss #
            self.criterionL1 = F.smooth_l1_loss #added on 24/02/01
            self.criterionL2 = torch.nn.MSELoss() #test on 24/01/31
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_A_a.parameters(), self.netD_B_a.parameters()),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netLD_A.parameters(), self.netLD_B.parameters(), self.netGD_A.parameters(), self.netGD_B.parameters(),self.netLD_A_a.parameters(), self.netLD_B_a.parameters(), self.netGD_A_a.parameters(), self.netGD_B_a.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    
    def set_input(self, input, input_u):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input_u['A' if AtoB else 'B'].to(self.device)
        self.real_B = input_u['B' if AtoB else 'A'].to(self.device)

        self.real_A_a = input['A' if AtoB else 'B'].to(self.device) # aligned image A
        self.real_B_a = input['B' if AtoB else 'A'].to(self.device) # aligned image B

        self.image_paths = input_u['A_paths' if AtoB else 'B_paths'] # It's ok

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        self.fake_B_a = self.netG_A(self.real_A_a)  # G_A(A_a)
        # self.rec_A_a = self.netG_B(self.fake_B_a)  # G_B(G_A(A))
        self.fake_A_a = self.netG_B(self.real_B_a)  # G_B(B_a)
        # self.rec_B_a = self.netG_A(self.fake_A_a)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_LD_basic(self, netLD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netLD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_LD.backward() to calculate the gradients.
        """
        # Real
        pred_real = netLD(real)
        loss_LD_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netLD(fake.detach())
        loss_LD_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_LD = (loss_LD_real + loss_LD_fake) * 0.5
        loss_LD.backward()
        return loss_LD
    
    def backward_GD_basic(self, netGD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netGD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_GD.backward() to calculate the gradients.
        """
        # Real
        pred_real = netGD(real)
        loss_GD_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netGD(fake.detach())
        loss_GD_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_GD = (loss_GD_real + loss_GD_fake) * 0.5
        loss_GD.backward()
        return loss_GD
    
    def backward_LD_A(self):
        """Calculate GAN loss for discriminator LD_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_LD_A = self.backward_LD_basic(self.netLD_A, self.real_B, fake_B)

    def backward_LD_B(self):
        """Calculate GAN loss for discriminator LD_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_LD_B = self.backward_LD_basic(self.netLD_B, self.real_A, fake_A)

    def backward_GD_A(self):
        """Calculate GAN loss for discriminator GD_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_GD_A = self.backward_GD_basic(self.netGD_A, self.real_B, fake_B)

    def backward_GD_B(self):
        """Calculate GAN loss for discriminator GD_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_GD_B = self.backward_GD_basic(self.netGD_B, self.real_A, fake_A)

    def backward_LD_A_a(self):
        """Calculate GAN loss for discriminator LD_A_aligned"""
        fake_B_a = self.fake_B_pool_a.query(self.fake_B_a)
        #self.loss_D_A_a = self.backward_D_basic(self.netD_A, self.real_B_a, fake_B_a)
        self.loss_LD_A_a = self.backward_LD_basic(self.netLD_A_a, self.real_B_a, fake_B_a)

    def backward_LD_B_a(self):
        """Calculate GAN loss for discriminator LD_B_aligned"""
        fake_A_a = self.fake_A_pool_a.query(self.fake_A_a)
        self.loss_LD_B_a = self.backward_LD_basic(self.netLD_B_a, self.real_A_a, fake_A_a)
    
    def backward_GD_A_a(self):
        """Calculate GAN loss for discriminator GD_A_aligned"""
        fake_B_a = self.fake_B_pool_a.query(self.fake_B_a)
        #self.loss_D_A_a = self.backward_D_basic(self.netD_A, self.real_B_a, fake_B_a)
        self.loss_GD_A_a = self.backward_GD_basic(self.netGD_A_a, self.real_B_a, fake_B_a)

    def backward_GD_B_a(self):
        """Calculate GAN loss for discriminator GD_B_aligned"""
        fake_A_a = self.fake_A_pool_a.query(self.fake_A_a)
        self.loss_GD_B_a = self.backward_GD_basic(self.netGD_B_a, self.real_A_a, fake_A_a)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_L1 = self.opt.lambda_L1
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss LD_A(G_A(A))
        self.loss_G_LA = self.criterionGAN(self.netLD_A(self.fake_B), True)
        # GAN loss LD_B(G_B(B))
        self.loss_G_LB = self.criterionGAN(self.netLD_B(self.fake_A), True)
        # GAN loss GD_A(G_A(A))
        self.loss_G_GA = self.criterionGAN(self.netGD_A(self.fake_B), True)
        # GAN loss GD_B(G_B(B))
        self.loss_G_GB = self.criterionGAN(self.netGD_B(self.fake_A), True)
        # GAN loss LD_A(G_A(A_a))
        self.loss_G_LA_a = self.criterionGAN(self.netLD_A_a(self.fake_B_a), True)
        # GAN loss LD_B(G_B(B_a))
        self.loss_G_LB_a = self.criterionGAN(self.netLD_B_a(self.fake_A_a), True)
        # GAN loss GD_A(G_A(A_a))
        self.loss_G_GA_a = self.criterionGAN(self.netGD_A_a(self.fake_B_a), True)
        # GAN loss GD_B(G_B(B_a))
        self.loss_G_GB_a = self.criterionGAN(self.netGD_B_a(self.fake_A_a), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Semi => pix2pix, G(A) = B, G(B) = A
        self.loss_G_A_L1 = self.criterionL1(self.fake_B_a, self.real_B_a) * lambda_L1
        self.loss_G_B_L1 = self.criterionL1(self.fake_A_a, self.real_A_a) * lambda_L1
        # self.loss_G_A_L1 = self.criterionL2(self.fake_B_a, self.real_B_a) * lambda_L1
        # self.loss_G_B_L1 = self.criterionL2(self.fake_A_a, self.real_A_a) * lambda_L1
        #test on 2024/1/31
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_LA + self.loss_G_LB + self.loss_G_GA + self.loss_G_GB + self.loss_G_LA_a + self.loss_G_LB_a + self.loss_G_GA_a + self.loss_G_GB_a + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_A_L1 + self.loss_G_B_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netLD_A, self.netLD_B, self.netGD_A, self.netGD_B, self.netLD_A_a, self.netLD_B_a, self.netGD_A_a, self.netGD_B_a], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        # self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        # self.backward_D_A()  # calculate gradients for D_A
        # self.backward_D_B()  # calculate graidents for D_B
        # self.optimizer_D.step()  # update D_A and D_B's weights

        # D_A, D_B, D_A_a, D_B_a
        self.set_requires_grad([self.netLD_A, self.netLD_B, self.netGD_A, self.netGD_B,self.netLD_A_a, self.netLD_B_a, self.netGD_A_a, self.netGD_B_a], True) # They could be separated
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_LD_A()      # calculate gradients for LD_A
        self.backward_LD_B()      # calculate graidents for LD_B
        self.backward_GD_A()      # calculate gradients for GD_A
        self.backward_GD_B()      # calculate graidents for GD_B
        self.backward_LD_A_a()      # calculate gradients for LD_A_a
        self.backward_LD_B_a()      # calculate graidents for LD_B_a
        self.backward_GD_A_a()      # calculate gradients for GD_A_a
        self.backward_GD_B_a()      # calculate graidents for GD_B_a
        self.optimizer_D.step()  # update D_A and D_B's weights
