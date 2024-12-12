/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package flinkClassifiersTesting;

import flinkClassifiersTesting.processors.factory.FlinkProcessFactory;
import flinkClassifiersTesting.processors.factory.arf.AdaptiveRandomForestClassifierParams;
import flinkClassifiersTesting.processors.factory.arf.AdaptiveRandomForestProcessFactory;
import flinkClassifiersTesting.processors.factory.atnn.AtnnClassifierParams;
import flinkClassifiersTesting.processors.factory.atnn.AtnnProcessFactory;
import flinkClassifiersTesting.processors.factory.cand.CandClassifierParams;
import flinkClassifiersTesting.processors.factory.cand.CandProcessFactory;
import flinkClassifiersTesting.processors.factory.vfdt.VfdtClassifierParams;
import flinkClassifiersTesting.processors.factory.vfdt.VfdtProcessFactory;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;

public class DataStreamJob {
    private static String getBaseDirectory() throws URISyntaxException {
        URI jarUri = DataStreamJob.class
                .getProtectionDomain()
                .getCodeSource()
                .getLocation()
                .toURI();
        File file = new File(jarUri);

        File parentDirectory = file.getParentFile();

        File parentOfParentDirectory = parentDirectory.getParentFile();

        return parentOfParentDirectory.getAbsolutePath();
    }

    public static void main(String[] args) throws Exception {
        String basePath = getBaseDirectory();
//        String dataset = "elec-maly";
//        String dataset = "elec";
//        String dataset = "sea_abr";
//        String dataset = "sea_grad";
//        String dataset = "mnist_grad";
//        String dataset = "elec-malutki";
//        String dataset = "mnist_grad_mniejszy";
//        String dataset = "mnist_grad_malutki";
        String dataset = "mnist_grad_powolny";
//        String dataset = "sea_grad";
        String datasetPath = basePath + "/datasets/" + dataset + ".csv";
//        long bootstrapSamplesLimit = 500L;
        long bootstrapSamplesLimit = 0L;

//        List<VfdtClassifierParams> vfdtParams = List.of(new VfdtClassifierParams(0.2, 0.1, 10));
//        FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, VfdtProcessFactory.vfdt(vfdtParams));

//        List<AdaptiveRandomForestClassifierParams> arfParams = List.of(
//                new AdaptiveRandomForestClassifierParams(30, 6.0f, 60)
//        );
//        FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, AdaptiveRandomForestProcessFactory.arf(arfParams));
//
        List<AtnnClassifierParams> atnnParams = List.of(
                new AtnnClassifierParams()
        );
        FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, AtnnProcessFactory.atnn(atnnParams));

//        List<CandClassifierParams> candParams = List.of(
//                new CandClassifierParams(CandClassifierParams.PoolSize.P30, 10, false)
//        );
//        FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, CandProcessFactory.cand(candParams));

    }
}
