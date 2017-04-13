/**
* =============================================================================
* Federal University of Rio Grande do Sul (UFRGS)
* Connectionist Artificial Intelligence Laboratory (LIAC)
* Edigleison F. Carvalho - edigleison.carvalho@inf.ufrgs.br
* =============================================================================
* Copyright (c) 2012 Edigleison F. Carvalho, edigleison.carvalho at gmail dot com
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy 
* of this software and associated documentation files (the "Software"), to deal 
* in the Software without restriction, including without limitation the rights 
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
* copies of the Software, and to permit persons to whom the Software is 
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in 
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
* SOFTWARE.
* =============================================================================
*/

package liac.igmn.core;
import java.util.ArrayList;
import java.util.List;

import liac.igmn.util.MatrixUtil;

import org.ejml.simple.SimpleMatrix;


public class IGMN
{
	/**
	 * Armazena a probabilidade a priori de cada componente numa matriz coluna
	 */
	protected SimpleMatrix priors;
	/**
	 * Armazena os vetores media de cada componente 
	 */
	protected List<SimpleMatrix> means;
	/**
	 * Armazena a matriz de covariancia de cada componente
	 */
	protected List<SimpleMatrix> covs;
	/**
	 * Armazena a soma das probabilidades a posteriori de cada componente numa matriz coluna
	 */
	protected SimpleMatrix sps;
	/**
	 * Armazena a verossimilhanca <p(x|j)> do ultimo vetor de entrada para cada componente numa matriz coluna
	 */
	protected SimpleMatrix like;
	/** 
	 * Armazena a probabilidade a posteriori <p(j|x)> do ultimo vetor de entrada para cada componente numa matriz coluna 
	 */
	protected SimpleMatrix post;
	/**
	 * Armazena a idade de cada componente numa matriz coluna
	 */
	protected SimpleMatrix vs;
	/**
	 * Armazena a dimensao do vetor de entrada
	 */
	protected int dimension;
	/**
	 * Armazena o numero de componentes 
	 */
	protected int size;
	/**
	 * Armazena o range do vetor de entrada
	 */
	protected SimpleMatrix dataRange;
	/**
	 * Parametro de ajuste do tamanho inicial da matriz de covariancia
	 */
	protected double delta;
	/**
	 * Parametro de ajuste da distancia minima aceitavel pelo componente
	 */
	protected double tau;
	/**
	 * Parametro de ajuste da ativacao minima de um componente para nao ser considerado como spurious
	 */
	protected double spMin;
	/**
	 * Parametro de ajuste da idade minima de um componente para poder ser detectado como spurious
	 */
	protected double vMin;
	/**
	 * Constante para definir o menor valor da verossimilhanca
	 */
	protected float eta;
	
	public IGMN(SimpleMatrix dataRange, double tau, double delta, double spMin, double vMin)
	{
		this.dataRange = dataRange;
		this.dimension = dataRange.getNumElements();
		this.size = 0;
		this.priors = new SimpleMatrix(0, 0);
		this.means = new ArrayList<SimpleMatrix>();
		this.covs = new ArrayList<SimpleMatrix>();
		this.sps = new SimpleMatrix(0, 0);
		this.like = new SimpleMatrix(0, 0);
		this.post = new SimpleMatrix(0, 0);
		this.vs = new SimpleMatrix(0, 0);
		this.delta = delta;
		this.tau = tau;
		this.spMin = spMin;
		this.vMin = vMin;
		this.eta = Float.MIN_VALUE;
	}
	
	public IGMN(SimpleMatrix dataRange, double tau, double delta)
	{
		this(dataRange, tau, delta, dataRange.getNumElements() + 1, 2 * dataRange.getNumElements());
	}
	
	/**
	 * Algoritmo de aprendizagem da rede IGMN
	 * 
	 * @param x vetor a ser utilizado no aprendizado
	 */
	public void learn(SimpleMatrix x)
	{
		computeLikelihood(x);
		if (!hasAcceptableDistribution())
		{
			addComponent(x);
			this.like.getMatrix().reshape(size, 1, true);
			int i = size - 1;
			this.like.set(i, 0, mvnpdf(x, means.get(i), covs.get(i)) + this.eta);
			updatePriors();
		}       
		computePosterior();
		incrementalEstimation(x);
		updatePriors();
		removeSpuriousComponents();
	}
	
	/**
	 * Realiza call para rede
	 * 
	 * @param x vetor de entrada
	 */
	public void call(SimpleMatrix x)
	{
		computeLikelihood(x);
		computePosterior();
	}
	
	/**
	 * Executa o algoritmo recall da IGMN
	 * 
	 * @param x vetor de entrada
	 * @return vetor resultante do recall
	 */
	public SimpleMatrix recall(SimpleMatrix x)
	{
		int alpha = x.getNumElements();
		int beta = dimension - alpha;
		
		SimpleMatrix pajs = new SimpleMatrix(size, 1);
		List<SimpleMatrix> xm = new ArrayList<SimpleMatrix>();
		
		for(int i = 0; i < size; i++)
		{
			SimpleMatrix cov = covs.get(i);
			SimpleMatrix covA = cov.extractMatrix(0, alpha, 0, alpha);
//			SimpleMatrix covB = cov.extractMatrix(alpha, alpha+beta, alpha, alpha+beta);
			SimpleMatrix covAB = cov.extractMatrix(0, alpha, alpha, alpha+beta);
			
			SimpleMatrix mean = means.get(i);
			SimpleMatrix meanA = mean.extractMatrix(0, alpha, 0, 1);
			SimpleMatrix meanB = mean.extractMatrix(alpha, alpha+beta, 0, 1);
			
			pajs.set(i, 0, mvnpdf(x, meanA, covA) + eta);
			SimpleMatrix x_ = meanB.plus(covAB.transpose().mult(covA.invert()).mult(x.minus(meanA)));
			xm.add(x_);
		}
		
		pajs = pajs.divide(pajs.elementSum());
		SimpleMatrix result = new SimpleMatrix(beta, 1);
		for (int i = 0; i < xm.size(); i++)
			result = result.plus(xm.get(i).scale(pajs.get(i)));
		
		return result;
	}
	
	/**
	 * Calcula a verossimilhanca para cada componente <p(x|j)>
	 * 
	 * @param x vetor de entrada
	 */
	public void computeLikelihood(SimpleMatrix x)
	{
		this.like = new SimpleMatrix(size, 1);
		for (int i = 0; i < size; i++)
			this.like.set(i, 0, mvnpdf(x, means.get(i), covs.get(i)) + eta);
	}
	
	/**
	 * Calcula a probabilidade a posteriori para cada componente <p(j|x)>
	 */
	public void computePosterior()
	{
		SimpleMatrix density = new SimpleMatrix(size, 1);
		
		for (int i = 0; i < size; i++)
			density.set(i, 0, this.like.get(i) * this.priors.get(i));
		
		this.post = density.divide(density.elementSum());
	}
	
	/**
	 * Atualiza as probabilidades a priori de cada componente
	 */
	public void updatePriors()
	{
		double spSum = sps.elementSum();
        priors = sps.divide(spSum);
	}
	
	/**
	 * Atualiza os parametros idade, acumulador de posteriori, media e matriz de covariancia
	 * 
	 * @param x vetor de entrada
	 */
	public void incrementalEstimation(SimpleMatrix x)
	{
		for (int i = 0; i < size; i++)
		{
			this.vs.set(i, this.vs.get(i) + 1);
			
			this.sps.set(i, this.sps.get(i) + this.post.get(i));
			
			SimpleMatrix oldmeans = this.means.get(i).copy();

			double w = this.post.get(i) / this.sps.get(i);
			SimpleMatrix diff = x.minus(oldmeans).scale(w);
			this.means.set(i, oldmeans.plus(diff));

			diff = this.means.get(i).minus(oldmeans);
			SimpleMatrix diff2 = x.minus(this.means.get(i));

			SimpleMatrix cov = this.covs.get(i);
			SimpleMatrix newCov = cov.minus(diff.mult(diff.transpose())).plus(diff2.mult(diff2.transpose()).minus(cov).scale(w));
			this.covs.set(i, newCov);
		}
	}
	
	/**
	 * Adiciona um novo componente na IGMN
	 * 
	 * @param x vetor que sera o centro do novo componente
	 */
	public void addComponent(SimpleMatrix x)
	{
		this.size += 1;
		this.priors.getMatrix().reshape(size, 1, true);
		this.priors.set(size - 1, 0, 1);
		this.means.add(new SimpleMatrix(x));
		SimpleMatrix newCov = new SimpleMatrix(dataRange.scale(delta));
		this.covs.add(MatrixUtil.diag(newCov.elementMult(newCov)));
		this.sps.getMatrix().reshape(size, 1, true);
		this.sps.set(size - 1, 0, 1);
		this.vs.getMatrix().reshape(size, 1, true);
		this.vs.set(size - 1, 0, 1);
	}
	
	/**
	 * Remove componentes que sao considerados ruidosos.
	 * O componente e removido caso sua idade seja maior que a idade minima <vMin>
	 * e se sua ativacao for menor que a ativacao minima <spMin>
	 */
	public void removeSpuriousComponents()
	{
		for(int i = size - 1; i >= 0; i--)
		{
			if (vs.get(i) > vMin && sps.get(i) < spMin)
			{
		        MatrixUtil.removeElement(vs, i);
		        MatrixUtil.removeElement(sps, i);
		        MatrixUtil.removeElement(priors, i);
				means.remove(i);
				covs.remove(i);
				size -= 1;
			}
		}
	}
	
	/**
	 * 
	 * @return <true> se o vetor de entrada tem verossimilhanca minima,
	 * determinado pelo parametro <tau>, para algum componente,
	 * <false> caso contrario 
	 */
	public boolean hasAcceptableDistribution()
	{
		for (int i = 0; i < size; i++)
		{
			double den = Math.pow(2 * Math.PI, dimension / 2.0) * Math.sqrt(covs.get(i).determinant());
			double min = tau / den;
			if (like.get(i) >= min)
				return true;
		}
		
        return false;
	}
	
	/**
	 * Realiza treinamento a partir de um conjunto de dados, onde
	 * cada instancia e uma coluna da matriz
	 * 
	 * @param dataset o conjunto de treinamento
	 */
	public void train(SimpleMatrix dataset)
	{
		for(int i = 0; i < dataset.numCols(); i++)
			learn(dataset.extractVector(false, i)); 
	}
	
	/**
	 * Classifica um vetor de entrada 
	 * @param x vetor de entrada
	 * @return vetor referente a classificacao do vetor de entrada
	 */
	public SimpleMatrix classify(SimpleMatrix x)
	{
		SimpleMatrix out = recall(x);
        int i = MatrixUtil.maxElementIndex(out);
        
        SimpleMatrix y = new SimpleMatrix(1, this.dimension - x.getNumElements());
        y.set(i, 1);
        
        return y;
	}
	
	/**
	 * Reinicia a rede
	 */
	public void reset()
	{
		this.size = 0;
		this.priors = new SimpleMatrix(0, 0);
		this.means = new ArrayList<SimpleMatrix>();
		this.covs = new ArrayList<SimpleMatrix>();
		this.sps = new SimpleMatrix(0, 0);
		this.like = new SimpleMatrix(0, 0);
		this.post = new SimpleMatrix(0, 0);
		this.vs = new SimpleMatrix(0, 0);
	}
	
	public SimpleMatrix getPriors()
	{
		return priors;
	}

	public List<SimpleMatrix> getMeans()
	{
		return means;
	}

	public List<SimpleMatrix> getCovs()
	{
		return covs;
	}

	public SimpleMatrix getSps()
	{
		return sps;
	}

	public SimpleMatrix getLike()
	{
		return like;
	}

	public SimpleMatrix getPost()
	{
		return post;
	}

	public SimpleMatrix getVs()
	{
		return vs;
	}

	public int getDimension()
	{
		return dimension;
	}

	public int getSize()
	{
		return size;
	}

	public SimpleMatrix getDataRange()
	{
		return dataRange;
	}

	public double getDelta()
	{
		return delta;
	}

	public double getTau()
	{
		return tau;
	}

	public double getSpMin()
	{
		return spMin;
	}

	public double getvMin()
	{
		return vMin;
	}

	/**
	 * Calcula a funcao de densidade de probabilidade multivariada (multivariate probability density function)
	 * 
	 * @param x vetor de entrada
	 * @param u vetor media
	 * @param cov matriz de covariancia
	 * @return a densidade de probabilidade
	 */
	private double mvnpdf(SimpleMatrix x, SimpleMatrix u, SimpleMatrix cov)
	{
		double dimension = x.getNumElements();
		SimpleMatrix distance = x.minus(u);
		
		double pdf = Math.exp(-0.5 * distance.transpose().dot(cov.invert().mult(distance)))
				/ (Math.pow(2 * Math.PI, dimension / 2.0) * Math.sqrt(cov.determinant()));

		pdf = Double.isNaN(pdf) ? 0 : pdf;
		
		return pdf;
	}
}
